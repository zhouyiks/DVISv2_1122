import random

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized
from .utils.ctvis_loss import loss_reid

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class ViHubCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, num_new_ins):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.num_new_ins = num_new_ins

    def loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = []
        target_classes = []
        for output_i, target_i, indice_i in zip(outputs, targets, indices):
            pred_logits = output_i["pred_logits"][0] # q, k+1
            tgt_classes_o = target_i[0]["labels"]    # ntgt,
            valid_inst = target_i[0]["valid_inst"][indice_i[0][1]]  # ntgt,
            if 'disappear_tgt_id' not in output_i:
                print(output_i.keys())
                exit(0)
            valid_inst[indice_i[0][1] == output_i["disappear_tgt_id"]] = False  # modeling disappear

            src_idx, tgt_idx = indice_i[0][0][valid_inst], indice_i[0][1][valid_inst]
            assert len(src_idx) == torch.sum(valid_inst)

            tgt_classes = torch.full(
                pred_logits.shape[:1], self.num_classes, dtype=torch.int64, device=pred_logits.device
            )
            tgt_classes[src_idx] = tgt_classes_o[tgt_idx]

            src_logits.append(pred_logits)
            target_classes.append(tgt_classes)

        src_logits = torch.cat(src_logits, dim=0).unsqueeze(0) # b, q, k+1
        target_classes = torch.cat(target_classes, dim=0).unsqueeze(0) # b, q

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_disappear(self, outputs, targets, indices, num_masks):
        src_logits = []
        target_classes = []
        for output_i, target_i in zip(outputs, targets):
            if output_i["disappear_logits"] is None:
                continue

            pred_logits = output_i["disappear_logits"][0]  # q, k+1
            tgt_classes = torch.full(
                pred_logits.shape[:1], self.num_classes, dtype=torch.int64, device=pred_logits.device
            )

            src_logits.append(pred_logits)
            target_classes.append(tgt_classes)

        if len(src_logits) == 0:
            loss_ce = outputs[0]["disappear_embeds"].sum() * 0.0
        else:
            src_logits = torch.cat(src_logits, dim=0).unsqueeze(0)  # b, q, k+1
            target_classes = torch.cat(target_classes, dim=0).unsqueeze(0)  # b, q
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        losses = {"loss_disappear": loss_ce}

        return losses

    def loss_valid(self, outputs, targets, indices, num_masks):
        src_scores = []
        target_labels = []
        for output_i, target_i, indice_i in zip(outputs, targets, indices):
            pred_scores = output_i["pred_valid"][0] # nq, 1
            tgt_labels = torch.full(pred_scores.shape[:1], 0).to(pred_scores)

            valid_inst = target_i[0]["valid_inst"]
            src_idx = indice_i[0][0][valid_inst]
            tgt_labels[src_idx] = 1.

            src_scores.append(pred_scores)
            target_labels.append(tgt_labels)

        src_scores = torch.cat(src_scores, dim=0)
        target_labels = torch.cat(target_labels, dim=0)[:, None]

        loss_bce = F.binary_cross_entropy_with_logits(src_scores, target_labels)
        losses = {"loss_bce": loss_bce}

        return losses

    def loss_cl(self, outputs, targets, indices, num_masks):
        contrastive_items = []

        use_disappear_embed_count = 0
        for output_i, target_i, indice_i in zip(outputs, targets, indices):
            reid_output_list = output_i["reid_outputs"]
            use_disappear_embed_count += output_i["use_disappear_embed_count"]
            for reid_out in reid_output_list:
                anchor_embedding = reid_out["anchor_embedding"].unsqueeze(0)
                positive_embedding = reid_out["positive_embedding"]
                if positive_embedding.ndim == 1:
                    positive_embedding = positive_embedding.unsqueeze(0)
                negative_embedding = reid_out["negative_embedding"]

                num_positive = positive_embedding.shape[0]
                pos_neg_embedding = torch.cat([positive_embedding, negative_embedding], dim=0)
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0], ), dtype=torch.int64)
                pos_neg_label[:num_positive] = 1

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embedding])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embedding, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

        if use_disappear_embed_count == 0:
            if len(contrastive_items) == 0:
                outputs[0]["reid_embeds"] = outputs[0]["reid_embeds"] + 0.0 * outputs[0]["disappear_embeds"].sum()
            else:
                contrastive_items[0]["cosine_similarity"] = (contrastive_items[0]["cosine_similarity"]
                                                             + 0.0 * outputs[0]["disappear_embeds"].sum())

        losses = loss_reid(contrastive_items, outputs[0])

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        src_masks = []
        target_masks = []
        for output_i, target_i, indice_i in zip(outputs, targets, indices):
            valid_inst = target_i[0]["valid_inst"][indice_i[0][1]]
            valid_inst[indice_i[0][1] == output_i["disappear_tgt_id"]] = False  # modeling disappear
            src_idx, tgt_idx = indice_i[0][0][valid_inst], indice_i[0][1][valid_inst]

            pred_masks = output_i["pred_masks"][0][src_idx] # ntgt, h, w
            tgt_masks  = target_i[0]["masks"][tgt_idx]      # ntgt, h, w

            src_masks.append(pred_masks)
            target_masks.append(tgt_masks)
        src_masks = torch.cat(src_masks, dim=0)
        target_masks = torch.cat(target_masks, dim=0).to(src_masks)
        N = src_masks.shape[0]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        point_logits = point_logits.view(N, self.num_points)
        point_labels = point_labels.view(N, self.num_points)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "valid": self.loss_valid,
            "reid": self.loss_cl,
            "disappear": self.loss_disappear,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        assert len(outputs) == len(targets)

        indices = [out["indices"] for out in outputs]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = 0
        for _target, _output in zip(targets, outputs):
            valid_mask = _target[0]["valid_inst"]
            if _output["disappear_tgt_id"] > -1:
                num_masks += (valid_mask.sum() - 1)
            else:
                num_masks += valid_mask.sum()
            # num_masks += valid_mask.sum()
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs[0].values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == "reid" and not "reid_outputs" in outputs[0]:
                continue
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks)
            )

        if "aux_outputs" in outputs[0]:
            len_aux = len(outputs[0]["aux_outputs"])
            for i in range(len_aux):
                aux_outputs = [outputs_i["aux_outputs"][i] for outputs_i in outputs]

                for loss in self.losses:
                    if loss in ["reid", "disappear"]:
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "fq_outputs" in outputs[0]:
            fq_outputs = [outputs_i["fq_outputs"] for outputs_i in outputs]
            fq_indices = [fq_outputs_i["indices"] for fq_outputs_i in fq_outputs]
            for loss in self.losses:
                if loss == "reid" and not "reid_outputs" in fq_outputs[0]:
                    continue
                l_dict = self.get_loss(loss, fq_outputs, targets, fq_indices, num_masks)
                l_dict = {k + f"_fq": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
