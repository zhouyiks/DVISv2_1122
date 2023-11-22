# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_vihub_config(cfg):
    cfg.MODEL.VIDEO_HEAD = CN()
    cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS = 10
    cfg.MODEL.VIDEO_HEAD.VALID_SCORE_LOSS_WEIGHT = 2.
    cfg.MODEL.VIDEO_HEAD.EMBED_SIM_WEIGHT = 2.
    cfg.MODEL.VIDEO_HEAD.CONSECUTIVE_FRAMES_MATCHING_METRIC = "embeddings"
    cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD = 0.4
    cfg.MODEL.VIDEO_HEAD.VALID_SCORE_THRESHOLD = 0.3
    cfg.MODEL.VIDEO_HEAD.NOISE_FRAME_NUM = 1 # when sequence length less than this value, then filtering the sequence as noise
    cfg.MODEL.VIDEO_HEAD.TEMPORAL_SCORE_TYPE = "mean"
    cfg.MODEL.VIDEO_HEAD.MASK_NMS_THR = 0.6

    cfg.MODEL.VIDEO_HEAD.REID_WEIGHTS = 2.
    cfg.MODEL.VIDEO_HEAD.AUX_REID_WEIGHTS = 3.
    cfg.MODEL.VIDEO_HEAD.MATCH_SCORE_THR = 0.3

    cfg.MODEL.VIDEO_HEAD.NUM_REID_HEAD_LAYERS = 3
    cfg.MODEL.VIDEO_HEAD.USING_THR = False

