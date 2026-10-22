"""Baseline Faster R-CNN model used for comparison in the project."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops


# ----- Baseline ROI box head -----
# These classes reproduce the standard Faster R-CNN second-stage head.
class TwoMLPHead(nn.Module):
    """Two fully connected layers used by the Faster R-CNN ROI box head."""

    def __init__(self, in_channels: int, representation_size: int):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """Predict class scores and class-specific bounding-box adjustments."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            if list(x.shape[2:]) != [1, 1]:
                raise ValueError(f"x has wrong shape {list(x.shape)}, expected last dims [1,1]")
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


# ----- Baseline RPN anchors -----
# The baseline uses the generic torchvision Faster R-CNN anchor configuration.
def _default_anchorgen() -> AnchorGenerator:
    """Create the default Faster R-CNN anchors used by the baseline model."""
    # One anchor scale is assigned to each FPN level.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


def build_custom_model(
    num_classes: int = 2,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    min_size: int = 800,
    max_size: int = 1333,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
) -> GeneralizedRCNN:
    """Build the baseline Faster R-CNN with a ResNet-50 + FPN backbone."""

    # Use the same default number of trainable ResNet layers as torchvision.
    is_trained = pretrained_backbone
    trainable_backbone_layers = _validate_trainable_layers(
        is_trained, trainable_backbone_layers, max_value=5, default_value=3
    )

    # When ImageNet weights are used, FrozenBatchNorm2d follows torchvision's setup.
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if pretrained_backbone else nn.BatchNorm2d
    weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    backbone_body = resnet50(weights=weights_backbone, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone_body, trainable_backbone_layers)
    out_channels = backbone.out_channels  # ResNet-50 FPN produces 256-channel feature maps.

    # ----- Region Proposal Network -----
    # RPN behavior and thresholds are kept unchanged in the modified detector,
    # which makes the model files easier to compare. with the default anchor configuration.
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn = RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n=rpn_pre_nms_top_n,
        post_nms_top_n=rpn_post_nms_top_n,
        nms_thresh=0.7,
        score_thresh=0.0,
    )

    # Standard Faster R-CNN ROI pooling, classification, and box regression heads.
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(out_channels * (resolution ** 2), representation_size)
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    # RoIHeads keeps the standard ROI classification and Smooth L1 regression losses.
    roi_heads = RoIHeads(
        box_roi_pool,
        box_head,
        box_predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    )

    # Resize and normalize images before they enter the detector.
    # GeneralizedRCNNTransform performs image normalization/resizing before the backbone.
    transform = GeneralizedRCNNTransform(
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,
    )
    return GeneralizedRCNN(backbone, rpn, roi_heads, transform)
