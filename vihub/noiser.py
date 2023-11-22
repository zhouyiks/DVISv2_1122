import torch
import random
import numpy as np
from scipy.optimize import linear_sum_assignment


class Noiser:
    def __init__(self, noise_ratio=0.8, mode='none'):
        assert mode in ['none', 'rand_binary_fuse']
        self.mode= mode
        self.noise_ratio = noise_ratio

    def _wa_noise_forward(self, cur_embeds, matched_indices):
        # embeds (q, b, c), classes (q)
        indices = list(range(cur_embeds.shape[0]))
        np.random.shuffle(indices)

        while len(indices) < len(matched_indices):
            _indices = list(range(cur_embeds.shape[0]))
            np.random.shuffle(_indices)
            indices.extend(_indices)

        noise_init = cur_embeds[indices[:len(matched_indices)]]
        weight_ratio = torch.rand(noise_init.shape[0], 1, 1)
        weight_ratio = torch.where(weight_ratio > 0.5, weight_ratio - 0.5, weight_ratio)
        noise_init = noise_init * weight_ratio.to(cur_embeds) + cur_embeds[matched_indices] * (
                    1.0 - weight_ratio.to(cur_embeds))
        return noise_init

    def _rand_binary_fuse_noising(self, ref_embeds, cur_embeds, frame_info):
        # embeds (q, b, c)
        # if frame_info is not None and 'valid_mask' in frame_info:
        #     valid_cur_idx = torch.arange(0, cur_embeds.shape[0]).to("cuda")[frame_info["valid_mask"]]
        #     rand_idx_1 = torch.randint(low=0, high=len(valid_cur_idx), size=(ref_embeds.shape[0], )).to(valid_cur_idx)
        #     rand_idx_2 = torch.randint(low=0, high=len(valid_cur_idx), size=(ref_embeds.shape[0], )).to(valid_cur_idx)
        #     rand_cur_idx_1 = valid_cur_idx[rand_idx_1]
        #     rand_cur_idx_2 = valid_cur_idx[rand_idx_2]
        # else:
        #     rand_cur_idx_1 = torch.randint(low=0, high=cur_embeds.shape[0], size=(ref_embeds.shape[0], )).to("cuda")
        #     rand_cur_idx_2 = torch.randint(low=0, high=cur_embeds.shape[0], size=(ref_embeds.shape[0], )).to("cuda")
        #
        # weight_ratio = torch.rand(ref_embeds.shape[0], 1, 1).to("cuda")
        # noise_init = cur_embeds[rand_cur_idx_1] * weight_ratio + cur_embeds[rand_cur_idx_2] * (1 - weight_ratio)

        noise_init = frame_info["disappear_embeds"]

        return noise_init, self._wa_noise_forward(cur_embeds)

    def match_embeds(self, ref_embeds, cur_embeds):
        # embeds (q, b, c)
        ref_embeds, cur_embeds = ref_embeds.detach()[:, 0, :], cur_embeds.detach()[:, 0, :]
        ref_embeds = ref_embeds / (ref_embeds.norm(dim=1)[:, None] + 1e-6)
        cur_embeds = cur_embeds / (cur_embeds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(cur_embeds, ref_embeds.transpose(0, 1))
        C = 1 - cos_sim

        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)
        C = C.transpose(0, 1)
        closest_idx = torch.argmin(C, dim=1) # q'
        indices = linear_sum_assignment(C.cpu())
        closest_idx[torch.as_tensor(indices[0]).to("cuda")] = torch.as_tensor(indices[1]).to("cuda")

        return closest_idx

    def __call__(self, ref_embeds, cur_embeds, activate=False, frame_info=None):
        matched_indices = self.match_embeds(ref_embeds, cur_embeds)
        if activate and random.random() < self.noise_ratio:
            if self.mode == 'none':
                return matched_indices, cur_embeds[matched_indices]
            elif self.mode == 'rand_binary_fuse':
                return matched_indices, self._wa_noise_forward(cur_embeds, matched_indices)
            else:
                raise NotImplementedError
        else:
            return matched_indices, cur_embeds[matched_indices]
