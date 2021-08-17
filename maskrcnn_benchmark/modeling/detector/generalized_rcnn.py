# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)#feature extractor  backbone.py
        self.rpn = build_rpn(cfg) # rpn.py
        self.roi_heads = build_roi_heads(cfg) #roi_heads
        self.da_heads = build_da_heads(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:#model forward
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)#[1,3,608,1088]
        features = self.backbone(images.tensors)#features [1,1024,38,68]
        proposals, proposal_losses = self.rpn(images, features, targets)#proposals：[BoxList(num_boxes=1000, image_width=1066, image_height=600, mode=xyxy)]
        da_losses = {}
        if self.roi_heads:
            x, result, detector_losses, da_ins_feas, da_ins_labels = self.roi_heads(features, proposals, targets)#x:[1000, 2048, 7, 7], da_ins_feas:[1000, 2048, 7, 7]
            if self.da_heads:
                da_losses = self.da_heads(features, da_ins_feas, da_ins_labels, targets)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses

        return result
