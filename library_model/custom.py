import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms, roi_align

#BOXES-----------------
def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0, inter / union, inter.new_zeros(()))

def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, width - 1)
    boxes[:, 1] = boxes[:, 1].clamp(0, height - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, width - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, height - 1)
    return boxes

def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    ws = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    hs = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    keep = (ws >= min_size) & (hs >= min_size)
    return torch.where(keep)[0]

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=1)

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)


#BOXCODER-----------------
@dataclass
class BoxCoderConfig:
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    bbox_xform_clip: float = math.log(1000.0 / 16)

class BoxCoder:
    def __init__(self, cfg: BoxCoderConfig = BoxCoderConfig()):
        self.wx, self.wy, self.ww, self.wh = cfg.weights
        self.bbox_xform_clip = cfg.bbox_xform_clip

    def encode(self, reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        #reference_boxes: gt, proposals: anchors/proposals
        gt = xyxy_to_cxcywh(reference_boxes)
        pr = xyxy_to_cxcywh(proposals)
        dx = (gt[:, 0] - pr[:, 0]) / pr[:, 2]
        dy = (gt[:, 1] - pr[:, 1]) / pr[:, 3]
        dw = torch.log(gt[:, 2] / pr[:, 2])
        dh = torch.log(gt[:, 3] / pr[:, 3])
        return torch.stack([dx * self.wx, dy * self.wy, dw * self.ww, dh * self.wh], dim=1)

    def decode(self, rel_codes: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        pr = xyxy_to_cxcywh(boxes)
        dx, dy, dw, dh = rel_codes.unbind(dim=1)
        dx = dx / self.wx
        dy = dy / self.wy
        dw = dw / self.ww
        dh = dh / self.wh
        dw = dw.clamp(max=self.bbox_xform_clip)
        dh = dh.clamp(max=self.bbox_xform_clip)
        pred_cx = dx * pr[:, 2] + pr[:, 0]
        pred_cy = dy * pr[:, 3] + pr[:, 1]
        pred_w = torch.exp(dw) * pr[:, 2]
        pred_h = torch.exp(dh) * pr[:, 3]
        pred = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
        return cxcywh_to_xyxy(pred)

#MATCHER+SAMPLER-----------------
@dataclass
class MatcherConfig:
    high_threshold: float
    low_threshold: float
    allow_low_quality_matches: bool = True

class Matcher:
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, cfg: MatcherConfig):
        self.high_threshold = cfg.high_threshold
        self.low_threshold = cfg.low_threshold
        self.allow_low_quality_matches = cfg.allow_low_quality_matches

    def __call__(self, match_quality_matrix: torch.Tensor) -> torch.Tensor:
        #returns matched gt index for each prediction, or negative codes
        if match_quality_matrix.numel() == 0:
            return torch.empty((match_quality_matrix.shape[1],), dtype=torch.int64, device=match_quality_matrix.device)

        #for each prediction: best gt
        matched_vals, matches = match_quality_matrix.max(dim=0)
        below_low = matched_vals < self.low_threshold
        between = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low] = self.BELOW_LOW_THRESHOLD
        matches[between] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            #ensure each gt has at least one match (low-quality)
            highest_quality_foreach_gt, pred_inds = match_quality_matrix.max(dim=1)
            #pred_inds: for each gt, which prediction gets max IoU
            matches[pred_inds] = torch.arange(match_quality_matrix.shape[0], device=matches.device)

        return matches

@dataclass
class SamplerConfig:
    batch_size_per_image: int
    positive_fraction: float

class BalancedPositiveNegativeSampler:
    def __init__(self, cfg: SamplerConfig):
        self.batch_size_per_image = cfg.batch_size_per_image
        self.positive_fraction = cfg.positive_fraction

    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        #labels: -1 ignore, 0 neg, >0 pos
        positive = torch.where(labels > 0)[0]
        negative = torch.where(labels == 0)[0]
        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(negative.numel(), num_neg)
        perm1 = torch.randperm(positive.numel(), device=labels.device)[:num_pos] if num_pos > 0 else positive.new_empty((0,), dtype=torch.int64)
        perm2 = torch.randperm(negative.numel(), device=labels.device)[:num_neg] if num_neg > 0 else negative.new_empty((0,), dtype=torch.int64)
        pos_idx = positive[perm1] if num_pos > 0 else positive.new_empty((0,), dtype=torch.int64)
        neg_idx = negative[perm2] if num_neg > 0 else negative.new_empty((0,), dtype=torch.int64)
        return torch.cat([pos_idx, neg_idx], dim=0)

#TRANSFORM-----------------
@dataclass
class TransformConfig:
    min_size: int = 800
    max_size: int = 1333
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

class GeneralizedTransform(nn.Module):
    def __init__(self, cfg: TransformConfig):
        super().__init__()
        self.cfg = cfg
        mean = torch.tensor(cfg.image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(cfg.image_std, dtype=torch.float32).view(3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _get_scale(self, h: int, w: int) -> float:
        min_side = min(h, w)
        max_side = max(h, w)
        scale = float(self.cfg.min_size) / float(min_side)
        if max_side * scale > self.cfg.max_size:
            scale = float(self.cfg.max_size) / float(max_side)
        return scale

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def resize(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]]):
        _, h, w = image.shape
        scale = self._get_scale(h, w)
        if scale != 1.0:
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
            if target is not None and target["boxes"].numel() > 0:
                boxes = target["boxes"] * boxes.new_tensor([scale, scale, scale, scale]) if False else target["boxes"] * target["boxes"].new_tensor([scale, scale, scale, scale])
                target = dict(target)
                target["boxes"] = boxes
        return image, target, (h, w), (image.shape[1], image.shape[2])

    def batch_images(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        sizes = [(int(im.shape[1]), int(im.shape[2])) for im in images]
        max_h = max(h for h, w in sizes)
        max_w = max(w for h, w in sizes)
        batched = []
        for im in images:
            _, h, w = im.shape
            pad_bottom = max_h - h
            pad_right = max_w - w
            im = F.pad(im, (0, pad_right, 0, pad_bottom), value=0.0)
            batched.append(im)
        return torch.stack(batched, dim=0), sizes

    def postprocess(
        self,
        detections: List[Dict[str, torch.Tensor]],
        resized_sizes: List[Tuple[int, int]],
        original_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        out: List[Dict[str, torch.Tensor]] = []
        for det, (rh, rw), (oh, ow) in zip(detections, resized_sizes, original_sizes):
            if det["boxes"].numel() == 0:
                out.append(det)
                continue
            scale_y = float(oh) / float(rh)
            scale_x = float(ow) / float(rw)
            boxes = det["boxes"].clone()
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y
            boxes = clip_boxes_to_image(boxes, oh, ow)
            out.append({**det, "boxes": boxes})
        return out

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        normalized: List[torch.Tensor] = []
        new_targets = None if targets is None else []
        original_sizes: List[Tuple[int, int]] = []
        resized_sizes: List[Tuple[int, int]] = []

        for i, im in enumerate(images):
            t = None if targets is None else targets[i]
            im = self.normalize(im)
            im, t, orig, resized = self.resize(im, t)
            normalized.append(im)
            original_sizes.append(orig)
            resized_sizes.append(resized)
            if targets is not None:
                new_targets.append(t if t is not None else targets[i])

        batch, image_sizes = self.batch_images(normalized)
        return batch, image_sizes, new_targets, original_sizes, resized_sizes


#BACKBONE: ResNet50 C2-C5-----------------
class ResNet50C2C5(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet50(weights=weights)

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1  # C2
        self.layer2 = m.layer2  # C3
        self.layer3 = m.layer3  # C4
        self.layer4 = m.layer4  # C5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


#FPN-----------------
class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        #extra P6 from P5 (like torchvision for FasterRCNN-ResNet50-FPN)
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        for m in list(self.lateral_convs) + list(self.output_convs) + [self.p6]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #order: c2,c3,c4,c5 -> p2,p3,p4,p5
        xs = [feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
        lat = [l(x) for l, x in zip(self.lateral_convs, xs)]
        #top-down
        for i in range(len(lat) - 1, 0, -1):
            up = F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
            lat[i - 1] = lat[i - 1] + up
        outs = [o(x) for o, x in zip(self.output_convs, lat)]
        p2, p3, p4, p5 = outs
        p6 = self.p6(p5)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}

#ANCHOR GENERATOR(multi-level)-----------------
@dataclass
class AnchorGenConfig:
    #torchvision defaults for resnet50_fpn
    sizes: Tuple[Tuple[int, ...], ...] = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 5

class AnchorGenerator(nn.Module):
    def __init__(self, cfg: AnchorGenConfig):
        super().__init__()
        self.cfg = cfg
        self.num_levels = len(cfg.sizes)
        self.num_anchors_per_location = [len(s) * len(r) for s, r in zip(cfg.sizes, cfg.aspect_ratios)]

        #precompute base anchors per level (centered at 0,0 in cxcywh)
        self.register_buffer("_dummy", torch.zeros(1))  # just to get device easily
        self.base_anchors: List[torch.Tensor] = []
        for sizes, ratios in zip(cfg.sizes, cfg.aspect_ratios):
            anchors = []
            for s in sizes:
                area = float(s * s)
                for r in ratios:
                    w = math.sqrt(area / r)
                    h = w * r
                    anchors.append([0.0, 0.0, w, h])
            self.base_anchors.append(torch.tensor(anchors, dtype=torch.float32))

    def _grid_anchors(self, grid_size: Tuple[int, int], stride: Tuple[float, float], base_anchors: torch.Tensor, device) -> torch.Tensor:
        gh, gw = grid_size
        sy, sx = stride
        shifts_x = (torch.arange(gw, device=device) + 0.5) * sx
        shifts_y = (torch.arange(gh, device=device) + 0.5) * sy
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)  # [K,2]
        K = centers.shape[0]
        A = base_anchors.shape[0]
        base = base_anchors.to(device)
        cx = centers[:, 0].unsqueeze(1).expand(K, A)
        cy = centers[:, 1].unsqueeze(1).expand(K, A)
        w = base[:, 2].unsqueeze(0).expand(K, A)
        h = base[:, 3].unsqueeze(0).expand(K, A)

        anchors_cxcywh = torch.stack([cx, cy, w, h], dim=2).reshape(-1, 4)
        anchors = cxcywh_to_xyxy(anchors_cxcywh)
        return anchors

    def forward(
        self,
        features: List[torch.Tensor],
        image_size: Tuple[int, int],
    ) -> List[torch.Tensor]:
        #returns per-level anchors (same device as features)
        H, W = image_size
        device = features[0].device
        anchors_per_level: List[torch.Tensor] = []

        for lvl, feat in enumerate(features):
            _, _, fh, fw = feat.shape
            stride_y = float(H) / float(fh)
            stride_x = float(W) / float(fw)
            anchors = self._grid_anchors((fh, fw), (stride_y, stride_x), self.base_anchors[lvl], device)
            anchors = clip_boxes_to_image(anchors, H, W)
            anchors_per_level.append(anchors)

        return anchors_per_level

#RPN HEAD + RPN-----------------
class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors_per_location: List[int]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.ModuleList([nn.Conv2d(in_channels, a, 1) for a in num_anchors_per_location])
        self.bbox_pred = nn.ModuleList([nn.Conv2d(in_channels, a * 4, 1) for a in num_anchors_per_location])

        for m in [self.conv] + list(self.cls_logits) + list(self.bbox_pred):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        deltas: List[torch.Tensor] = []
        for i, x in enumerate(feats):
            t = F.relu(self.conv(x))
            logits.append(self.cls_logits[i](t))
            deltas.append(self.bbox_pred[i](t))
        return logits, deltas

@dataclass
class RPNConfig:
    fg_iou_thresh: float = 0.7
    bg_iou_thresh: float = 0.3
    batch_size_per_image: int = 256
    positive_fraction: float = 0.5
    pre_nms_top_n_train: int = 2000
    pre_nms_top_n_test: int = 1000
    post_nms_top_n_train: int = 1000
    post_nms_top_n_test: int = 1000
    nms_thresh: float = 0.7
    min_size: float = 1.0

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels: int, anchor_gen: AnchorGenerator, cfg: RPNConfig):
        super().__init__()
        self.anchor_gen = anchor_gen
        self.cfg = cfg
        self.head = RPNHead(in_channels, anchor_gen.num_anchors_per_location)
        self.box_coder = BoxCoder()
        self.matcher = Matcher(MatcherConfig(high_threshold=cfg.fg_iou_thresh, low_threshold=cfg.bg_iou_thresh))
        self.sampler = BalancedPositiveNegativeSampler(SamplerConfig(cfg.batch_size_per_image, cfg.positive_fraction))

    def _permute_and_flatten(self, layer: torch.Tensor, A: int, C: int) -> torch.Tensor:
        #[N, A*C, H, W] -> [N, H*W*A, C]
        N, _, H, W = layer.shape
        layer = layer.view(N, A, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2).reshape(N, -1, C)
        return layer

    def _concat_box_prediction_layers(
        self, objectness: List[torch.Tensor], pred_bbox_deltas: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obj_flattened = []
        deltas_flattened = []
        for obj, deltas in zip(objectness, pred_bbox_deltas):
            N, A, H, W = obj.shape
            obj_flattened.append(self._permute_and_flatten(obj, A, 1).squeeze(-1))
            deltas_flattened.append(self._permute_and_flatten(deltas, A, 4))

        return torch.cat(obj_flattened, dim=1), torch.cat(deltas_flattened, dim=1)

    def _get_top_n(self, is_training: bool) -> Tuple[int, int]:
        if is_training:
            return self.cfg.pre_nms_top_n_train, self.cfg.post_nms_top_n_train
        return self.cfg.pre_nms_top_n_test, self.cfg.post_nms_top_n_test

    def _filter_proposals(
        self,
        proposals: torch.Tensor,
        scores: torch.Tensor,
        image_size: Tuple[int, int],
        pre_nms_top_n: int,
        post_nms_top_n: int,
    ) -> torch.Tensor:
        H, W = image_size
        proposals = clip_boxes_to_image(proposals, H, W)
        keep = remove_small_boxes(proposals, self.cfg.min_size)
        proposals = proposals[keep]
        scores = scores[keep]
        num = min(pre_nms_top_n, scores.numel())
        if num <= 0:
            return proposals
        topk = torch.topk(scores, num).indices
        proposals = proposals[topk]
        scores = scores[topk]
        keep_n = nms(proposals, scores, self.cfg.nms_thresh)
        keep_n = keep_n[:post_nms_top_n]
        return proposals[keep_n]

    def assign_targets_to_anchors(
        self, anchors: torch.Tensor, gt_boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #labels: -1 ignore, 0 bg, 1 fg
        N = anchors.shape[0]
        if gt_boxes.numel() == 0:
            labels = anchors.new_zeros((N,), dtype=torch.int64)
            matched_gt = anchors.new_zeros((N, 4))
            return labels, matched_gt

        ious = box_iou(gt_boxes, anchors)  #[G, A]
        matched_idxs = self.matcher(ious)  [A]
        labels = anchors.new_full((N,), -1, dtype=torch.int64)
        bg = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        ignore = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        fg = matched_idxs >= 0
        labels[bg] = 0
        labels[ignore] = -1
        labels[fg] = 1

        matched_gt = gt_boxes[matched_idxs.clamp(min=0)]
        return labels, matched_gt

    def compute_loss(
        self,
        objectness: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        anchors: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        #per-image sampling
        device = objectness.device
        loss_obj = torch.tensor(0.0, device=device)
        loss_reg = torch.tensor(0.0, device=device)

        for i, tgt in enumerate(targets):
            gt_boxes = tgt["boxes"]
            labels, matched_gt = self.assign_targets_to_anchors(anchors[i], gt_boxes)
            sampled = self.sampler(labels)
            #objectness
            obj_i = objectness[i][sampled]
            lbl_i = labels[sampled].float()
            loss_obj = loss_obj + F.binary_cross_entropy_with_logits(obj_i, lbl_i, reduction="mean")
            # box regression only for positives
            pos = sampled[labels[sampled] == 1]
            if pos.numel() > 0:
                reg_targets = self.box_coder.encode(matched_gt[pos], anchors[i][pos])
                loss_reg = loss_reg + F.smooth_l1_loss(pred_bbox_deltas[i][pos], reg_targets, beta=1.0, reduction="mean")

        loss_obj = loss_obj / max(1, len(targets))
        loss_reg = loss_reg / max(1, len(targets))
        return {"rpn_objectness": loss_obj, "rpn_box_reg": loss_reg}

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        images_sizes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        feats_list = [features["p2"], features["p3"], features["p4"], features["p5"], features["p6"]]
        objectness_list, pred_bbox_deltas_list = self.head(feats_list)

        #reshape per level
        obj = []
        deltas = []
        for o, d, A in zip(objectness_list, pred_bbox_deltas_list, self.anchor_gen.num_anchors_per_location):
            obj.append(o)                       #[N,A,H,W]
            deltas.append(d)                    #[N,4A,H,W]

        objectness, pred_bbox_deltas = self._concat_box_prediction_layers(obj, deltas)  #[N,AllAnchors], [N,AllAnchors,4]

        #anchors per image: concat levels
        anchors_per_image: List[torch.Tensor] = []
        for i in range(len(images_sizes)):
            per_level = self.anchor_gen([f[i:i+1] for f in feats_list], images_sizes[i])  #list of [A_l,4]
            anchors_per_image.append(torch.cat(per_level, dim=0))

        #proposals
        is_training = targets is not None
        pre_nms_top_n, post_nms_top_n = self._get_top_n(is_training)

        proposals: List[torch.Tensor] = []
        for i in range(len(images_sizes)):
            anchors_i = anchors_per_image[i]
            deltas_i = pred_bbox_deltas[i]
            obj_i = objectness[i].sigmoid()

            props = self.box_coder.decode(deltas_i, anchors_i)
            props = self._filter_proposals(props, obj_i, images_sizes[i], pre_nms_top_n, post_nms_top_n)
            proposals.append(props.detach())  #critical for stability (esp. MPS)

        losses: Dict[str, torch.Tensor] = {}
        if targets is not None:
            losses = self.compute_loss(objectness, pred_bbox_deltas, anchors_per_image, targets)

        return proposals, losses


# ROI ALIGN (multi-scale) + ROI HEADS-----------------
@dataclass
class ROIConfig:
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.5
    batch_size_per_image: int = 512
    positive_fraction: float = 0.25
    bbox_reg_weights: Tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100

class TwoMLPHead(nn.Module):
    def __init__(self, in_channels: int, representation_size: int = 1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        for m in [self.fc6, self.fc7]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

def _proposal_level_assign(
    boxes: torch.Tensor,
    k_min: int = 2,
    k_max: int = 5,
    canonical_scale: int = 224,
    canonical_level: int = 4,
) -> torch.Tensor:
    #FPN paper rule
    eps = 1e-6
    box_sizes = torch.sqrt(box_area(boxes)).clamp(min=eps)
    level = torch.floor(canonical_level + torch.log2(box_sizes / canonical_scale + eps))
    level = level.clamp(min=k_min, max=k_max)
    return level.to(torch.int64)

class MultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size: int = 7, sampling_ratio: int = 2, featmap_names: Tuple[str, ...] = ("p2","p3","p4","p5")):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.featmap_names = featmap_names

    def forward(self, features: Dict[str, torch.Tensor], boxes: List[torch.Tensor], image_sizes: List[Tuple[int,int]]) -> torch.Tensor:
        #flatten all boxes with batch indices
        device = next(iter(features.values())).device
        rois = []
        levels = []
        for i, b in enumerate(boxes):
            if b.numel() == 0:
                continue
            lvl = _proposal_level_assign(b)  # in [2..5]
            batch_idx = torch.full((b.shape[0], 1), i, device=device, dtype=b.dtype)
            rois.append(torch.cat([batch_idx, b], dim=1))
            levels.append(lvl)
        if not rois:
            #return empty tensor [0, C, out, out]
            C = features[self.featmap_names[0]].shape[1]
            return torch.zeros((0, C, self.output_size, self.output_size), device=device)

        rois = torch.cat(rois, dim=0)
        levels = torch.cat(levels, dim=0)

        #do per-level roi_align then scatter back in correct order
        C = features[self.featmap_names[0]].shape[1]
        out = torch.zeros((rois.shape[0], C, self.output_size, self.output_size), device=device, dtype=features[self.featmap_names[0]].dtype)

        #compute spatial_scales from image_sizes vs feature map sizes (use 1st image size because padded batch is same, but per-feature map is consistent)
        #we assume features are computed from the padded batch; still, stride is stable
        #for each level name p2..p5 we approximate stride using first image size and feat size
        for name, k in zip(self.featmap_names, [2,3,4,5]):
            idx = torch.where(levels == k)[0]
            if idx.numel() == 0:
                continue
            feat = features[name]
            _, _, fh, fw = feat.shape
            H, W = image_sizes[0]
            stride_y = float(H) / float(fh)
            stride_x = float(W) / float(fw)
            #roi_align wants spatial_scale in x/y the same; in practice strides are square here
            spatial_scale = 1.0 / float(stride_x)

            pooled = roi_align(
                feat,
                rois[idx],
                output_size=(self.output_size, self.output_size),
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True,
            )
            out[idx] = pooled
        return out

class RoIHeads(nn.Module):
    def __init__(self, num_classes: int, cfg: ROIConfig):
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg
        self.box_coder = BoxCoder(BoxCoderConfig(weights=cfg.bbox_reg_weights))
        self.matcher = Matcher(MatcherConfig(high_threshold=cfg.fg_iou_thresh, low_threshold=cfg.bg_iou_thresh))
        self.sampler = BalancedPositiveNegativeSampler(SamplerConfig(cfg.batch_size_per_image, cfg.positive_fraction))
        self.pooler = MultiScaleRoIAlign(output_size=7, sampling_ratio=2, featmap_names=("p2","p3","p4","p5"))
        self.box_head = TwoMLPHead(in_channels=256 * 7 * 7, representation_size=1024)
        self.box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=num_classes)

    def assign_targets_to_proposals(self, proposals: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        N = proposals.shape[0]
        if gt_boxes.numel() == 0:
            labels = proposals.new_zeros((N,), dtype=torch.int64)
            matched_gt = proposals.new_zeros((N, 4))
            return labels, matched_gt

        ious = box_iou(gt_boxes, proposals)  # [G, P]
        matched_idxs = self.matcher(ious)    # [P]
        labels = proposals.new_full((N,), -1, dtype=torch.int64)
        bg = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        ignore = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        fg = matched_idxs >= 0
        labels[bg] = 0
        labels[ignore] = -1
        labels[fg] = gt_labels[matched_idxs[fg]]
        matched_gt = gt_boxes[matched_idxs.clamp(min=0)]
        return labels, matched_gt

    def subsample(self, labels: torch.Tensor) -> torch.Tensor: return self.sampler(labels)

    def fastrcnn_loss(self, class_logits: torch.Tensor, box_regression: torch.Tensor, labels: torch.Tensor, regression_targets: torch.Tensor):
        #classification
        loss_classifier = F.cross_entropy(class_logits, labels)

        #box regression only for positive labels
        sampled_pos = torch.where(labels > 0)[0]
        if sampled_pos.numel() > 0:
            #box_regression is [N, num_classes*4]
            box_regression = box_regression.reshape(-1, self.num_classes, 4)
            labels_pos = labels[sampled_pos]
            pred = box_regression[sampled_pos, labels_pos]
            loss_box_reg = F.smooth_l1_loss(pred, regression_targets[sampled_pos], beta=1.0, reduction="mean")
        else:
            loss_box_reg = loss_classifier.new_tensor(0.0)

        return loss_classifier, loss_box_reg

    def postprocess_detections(self, class_logits: torch.Tensor, box_regression: torch.Tensor, proposals: torch.Tensor, image_size: Tuple[int,int]):
        device = class_logits.device
        H, W = image_size
        scores = F.softmax(class_logits, dim=-1)
        box_regression = box_regression.reshape(-1, self.num_classes, 4)
        boxes_all = []
        scores_all = []
        labels_all = []
        for c in range(1, self.num_classes):
            cls_scores = scores[:, c]
            keep = cls_scores >= self.cfg.score_thresh
            if keep.sum() == 0:
                continue
            cls_scores = cls_scores[keep]
            cls_boxes = self.box_coder.decode(box_regression[keep, c], proposals[keep])
            cls_boxes = clip_boxes_to_image(cls_boxes, H, W)
            keep2 = remove_small_boxes(cls_boxes, 1.0)
            cls_boxes = cls_boxes[keep2]
            cls_scores = cls_scores[keep2]
            if cls_scores.numel() == 0:
                continue
            keep3 = nms(cls_boxes, cls_scores, self.cfg.nms_thresh)
            cls_boxes = cls_boxes[keep3]
            cls_scores = cls_scores[keep3]
            boxes_all.append(cls_boxes)
            scores_all.append(cls_scores)
            labels_all.append(torch.full((cls_boxes.shape[0],), c, dtype=torch.int64, device=device))

        if not boxes_all:
            return {
                "boxes": torch.zeros((0,4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
            }
        boxes = torch.cat(boxes_all, dim=0)
        scores = torch.cat(scores_all, dim=0)
        labels = torch.cat(labels_all, dim=0)

        #limit detections per image
        num = min(self.cfg.detections_per_img, scores.numel())
        topk = torch.topk(scores, num).indices
        return {"boxes": boxes[topk], "scores": scores[topk], "labels": labels[topk]}

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_sizes: List[Tuple[int,int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        if targets is not None:
            #add gt to proposals (torchvision does this)
            new_props: List[torch.Tensor] = []
            for props, tgt in zip(proposals, targets):
                gt = tgt["boxes"]
                if gt.numel() > 0:
                    props = torch.cat([props, gt], dim=0)
                new_props.append(props)
            proposals = new_props

        #pool
        box_features = self.pooler(features, proposals, image_sizes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        if targets is not None:
            #build labels/targets aligned to pooled proposals order
            labels_list = []
            reg_targets_list = []

            offset = 0
            for props, tgt in zip(proposals, targets):
                n = props.shape[0]
                gt_boxes = tgt["boxes"]
                gt_labels = tgt["labels"]

                labels, matched_gt = self.assign_targets_to_proposals(props, gt_boxes, gt_labels)
                sampled = self.subsample(labels)

                sampled_labels = labels[sampled]
                regression_targets = torch.zeros((n, 4), device=props.device, dtype=props.dtype)
                pos = torch.where(labels > 0)[0]
                if pos.numel() > 0:
                    regression_targets[pos] = self.box_coder.encode(matched_gt[pos], props[pos])

                #keep only sampled for loss
                labels_list.append(sampled_labels)
                reg_targets_list.append(regression_targets[sampled])

                #slice corresponding predictions
                offset += n

            labels_cat = torch.cat(labels_list, dim=0)
            reg_targets_cat = torch.cat(reg_targets_list, dim=0)

            #IMPORTANT: we must also select the same sampled indices in predictions.
            #easiest correct approach: re-run sampling indices with consistent proposal order.
            #to keep this file simple and correct, we reconstruct the sampled indices in prediction space:
            sampled_pred_indices = []
            base = 0
            for props, tgt in zip(proposals, targets):
                n = props.shape[0]
                gt_boxes = tgt["boxes"]
                gt_labels = tgt["labels"]
                labels, _ = self.assign_targets_to_proposals(props, gt_boxes, gt_labels)
                sampled = self.subsample(labels)
                sampled_pred_indices.append(base + sampled)
                base += n
            sampled_pred_indices = torch.cat(sampled_pred_indices, dim=0)

            cls_logits_s = class_logits[sampled_pred_indices]
            box_reg_s = box_regression[sampled_pred_indices]

            loss_classifier, loss_box_reg = self.fastrcnn_loss(cls_logits_s, box_reg_s, labels_cat, reg_targets_cat)
            losses = {"roi_classifier": loss_classifier, "roi_box_reg": loss_box_reg}
            return [], losses

        #inference: split predictions back per image
        detections: List[Dict[str, torch.Tensor]] = []
        start = 0
        for props, imsz in zip(proposals, image_sizes):
            n = props.shape[0]
            cls = class_logits[start:start+n]
            reg = box_regression[start:start+n]
            det = self.postprocess_detections(cls, reg, props, imsz)
            detections.append(det)
            start += n
        return detections, {}

#FASTER R-CNN (ours, FPN-based)-----------------
@dataclass
class ModelConfig:
    num_classes: int = 2
    pretrained_backbone: bool = True

class CustomFasterRCNNFPN(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.transform = GeneralizedTransform(TransformConfig())
        self.backbone = ResNet50C2C5(pretrained=cfg.pretrained_backbone)
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=256)
        anchor_gen = AnchorGenerator(AnchorGenConfig())
        self.rpn = RegionProposalNetwork(in_channels=256, anchor_gen=anchor_gen, cfg=RPNConfig())
        self.roi_heads = RoIHeads(num_classes=cfg.num_classes, cfg=ROIConfig())

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if targets is not None:
            #basic checks like torchvision does
            for t in targets:
                if "boxes" not in t or "labels" not in t:
                    raise ValueError("Each target must contain 'boxes' and 'labels'")
                if t["boxes"].numel() > 0:
                    if (t["boxes"][:, 2] <= t["boxes"][:, 0]).any() or (t["boxes"][:, 3] <= t["boxes"][:, 1]).any():
                        raise ValueError("All boxes must have x2>x1 and y2>y1")

        batch, image_sizes, targets, original_sizes, resized_sizes = self.transform(images, targets)

        #backbone + fpn
        feats_c = self.backbone(batch)
        feats = self.fpn(feats_c)
        proposals, rpn_losses = self.rpn(feats, image_sizes, targets)
        detections, roi_losses = self.roi_heads(feats, proposals, image_sizes, targets)
        if targets is not None:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        #map boxes back to original sizes like torchvision
        detections = self.transform.postprocess(detections, resized_sizes, original_sizes)
        return detections

def build_custom_model(num_classes: int = 2, pretrained_backbone: bool = True) -> CustomFasterRCNNFPN:
    return CustomFasterRCNNFPN(ModelConfig(num_classes=num_classes, pretrained_backbone=pretrained_backbone))