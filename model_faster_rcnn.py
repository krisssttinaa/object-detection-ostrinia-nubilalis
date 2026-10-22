from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops



def focal_loss(class_logits: torch.Tensor, labels: torch.Tensor, alpha_pos: float = 0.25, alpha_neg: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Binary/multiclass focal loss for the ROI classifier. Class 0 is background; >0 is foreground."""
    ce = F.cross_entropy(class_logits, labels, reduction="none")
    probs = F.softmax(class_logits, dim=1)
    pt = probs.gather(1, labels[:, None]).squeeze(1).clamp(min=1e-8, max=1.0)
    alpha = torch.where(
        labels > 0,
        torch.full_like(pt, alpha_pos),
        torch.full_like(pt, alpha_neg),
    )
    return (alpha * (1.0 - pt).pow(gamma) * ce).mean()


def complete_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU (CIoU) loss for aligned boxes in xyxy format."""
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    px1, py1, px2, py2 = pred_boxes.unbind(dim=1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(dim=1)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = pw * ph + tw * th - inter
    iou = inter / (union + eps)

    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    center_dist_sq = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    ex1 = torch.minimum(px1, tx1)
    ey1 = torch.minimum(py1, ty1)
    ex2 = torch.maximum(px2, tx2)
    ey2 = torch.maximum(py2, ty2)
    enclosing_diag_sq = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    v = (4.0 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    ciou = iou - center_dist_sq / enclosing_diag_sq - alpha * v
    return (1.0 - ciou).mean()


class OstriniaRoIHeads(RoIHeads):
    """RoIHeads with exactly two training-loss changes: Focal classification + CIoU box regression."""
    def __init__(self, *args, focal_alpha_pos: float = 0.25, focal_alpha_neg: float = 0.75, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_alpha_pos = focal_alpha_pos
        self.focal_alpha_neg = focal_alpha_neg
        self.focal_gamma = focal_gamma

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                if t["boxes"].dtype not in (torch.float, torch.double, torch.half):
                    raise TypeError(f"target boxes must be float type, got {t['boxes'].dtype}")
                if t["labels"].dtype != torch.int64:
                    raise TypeError(f"target labels must be int64, got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = regression_targets = matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            labels_cat = torch.cat(labels, dim=0)
            regression_targets_cat = torch.cat(regression_targets, dim=0)
            proposals_cat = torch.cat(proposals, dim=0)

            loss_classifier = focal_loss(
                class_logits, labels_cat,
                alpha_pos=self.focal_alpha_pos,
                alpha_neg=self.focal_alpha_neg,
                gamma=self.focal_gamma,
            )

            pos = torch.where(labels_cat > 0)[0]
            if pos.numel() > 0:
                box_regression_reshaped = box_regression.reshape(class_logits.shape[0], -1, 4)
                pred_deltas = box_regression_reshaped[pos, labels_cat[pos]]
                target_deltas = regression_targets_cat[pos]
                pos_proposals = proposals_cat[pos]
                pred_boxes = self.box_coder.decode_single(pred_deltas, pos_proposals)
                target_boxes = self.box_coder.decode_single(target_deltas, pos_proposals)
                loss_box_reg = complete_iou_loss(pred_boxes, target_boxes)
            else:
                loss_box_reg = box_regression.sum() * 0.0

            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, out_labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": out_labels[i], "scores": scores[i]})

        return result, losses

class TwoMLPHead(nn.Module):
    """standard torchvision-style box head for FPN Faster R-CNN"""
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
    """
      - scores: [N, num_classes]
      - bbox_deltas: [N, num_classes * 4] (class-specific regression)
    """
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

def _anchorgen(anchor_sizes=None, anchor_ratios=None) -> AnchorGenerator:
    if anchor_sizes is None:
        anchor_sizes = (32.0, 64.0, 128.0, 256.0, 512.0)
    if anchor_ratios is None:
        anchor_ratios = (0.5, 1.0, 2.0)
    sizes = tuple((float(x),) for x in anchor_sizes)
    ratios = (tuple(float(x) for x in anchor_ratios),) * len(sizes)
    return AnchorGenerator(sizes, ratios)

def build_custom_model(
    num_classes: int = 2,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    min_size: int = 800,
    max_size: int = 1333,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
    focal_alpha_pos: float = 0.25,
    focal_alpha_neg: float = 0.75,
    focal_gamma: float = 2.0,
    anchor_sizes=None,
    anchor_ratios=None,
    load_backbone_weights: bool = True,) -> GeneralizedRCNN:
    """this loads ImageNet weights for the backbone if pretrained_backbone=True"""

    #match torchvision’s behavior for trainable layers default
    is_trained = pretrained_backbone
    trainable_backbone_layers = _validate_trainable_layers(
        is_trained, trainable_backbone_layers, max_value=5, default_value=3)

    #torchvision uses FrozenBatchNorm2d when using pretrained backbone weights
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if pretrained_backbone else nn.BatchNorm2d
    weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone and load_backbone_weights else None
    backbone_body = resnet50(weights=weights_backbone, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone_body, trainable_backbone_layers)
    out_channels = backbone.out_channels  #256 for FPN

    #RPN
    rpn_anchor_generator = _anchorgen(anchor_sizes, anchor_ratios)
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
        score_thresh=0.0,)

    #ROI heads
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,)
    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(out_channels * (resolution ** 2), representation_size)
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    roi_heads = OstriniaRoIHeads(
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
        focal_alpha_pos=focal_alpha_pos,
        focal_alpha_neg=focal_alpha_neg,
        focal_gamma=focal_gamma,)
    transform = GeneralizedRCNNTransform(
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,)
    return GeneralizedRCNN(backbone, rpn, roi_heads, transform)