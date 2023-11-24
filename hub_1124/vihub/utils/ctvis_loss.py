import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def loss_reid(qd_items, outputs, reduce=False, name=None):
    contras_loss = 0
    aux_loss = 0

    num_qd_items = len(qd_items)
    if reduce:  # it seems worse when reduce is True
        num_qd_items = torch.as_tensor(
            [num_qd_items], dtype=torch.float, device=outputs['reid_embeds'].device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_qd_items)
        num_qd_items = torch.clamp(
            num_qd_items / get_world_size(), min=1).item()

    if len(qd_items) == 0:
        if name is not None:
            losses = {f'{name}_loss_reid': outputs[f'reid_{name}_embeds'].sum() * 0,
                      f'{name}_loss_aux_reid': outputs[f'reid_{name}_embeds'].sum() * 0}

        else:
            losses = {'loss_reid': outputs['reid_embeds'].sum() * 0,
                      'loss_aux_reid': outputs['reid_embeds'].sum() * 0}
        return losses

    for qd_item in qd_items:
        pred = qd_item['dot_product'].permute(1, 0)
        label = qd_item['label'].unsqueeze(0)
        # contrastive loss
        pos_inds = (label == 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])
        # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
        x = torch.nn.functional.pad(
            (_neg_expand - _pos_expand), (0, 1), "constant", 0)
        contras_loss += torch.logsumexp(x, dim=1)

        aux_pred = qd_item['cosine_similarity'].permute(1, 0)
        aux_label = qd_item['label'].unsqueeze(0)

        aux_loss += (torch.abs(aux_pred - aux_label) ** 2).mean()

    if name is not None:
        losses = {f'{name}_loss_reid': contras_loss.sum() / num_qd_items,
                  f'{name}_loss_aux_reid': aux_loss / num_qd_items}
    else:
        losses = {'loss_reid': contras_loss.sum() / num_qd_items,
                  'loss_aux_reid': aux_loss / num_qd_items}
    return losses