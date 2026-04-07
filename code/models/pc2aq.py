import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from models.backbone import build_backbone
from models.matcher import build_matcher
from models.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from models.transformer import build_transformer
import math
dependencies = ["torch", "torchvision"]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def pdist(x, y):
    """
    计算两组向量之间的欧氏距离（平方）
    Args:
        x: [N, D]
        y: [M, D]
    Returns:
        dist: [N, M] 欧氏距离的平方
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return dist


class PC2AQ(nn.Module):
    """
    小样本视频目标检测 DETR
    
    特点：
    1. 基于 DETR 进行目标检测（预测 bbox）
    2. 使用 ProtoNet 风格的距离度量进行分类（替代原来的分类头）
    3. 支持视频序列输入，包含时序建模能力
    4. 针对遥感视频小目标优化：多尺度特征、时序注意力、运动感知
    """
    
    def __init__(self, backbone, transformer, backbone_dim, num_classes, num_instances, aux_loss=False,
                 use_temporal_attention=True, num_frames=5):
        """
        Args:
            backbone: backbone 网络
            transformer: transformer 网络
            backbone_dim: backbone 输出维度
            num_classes: 类别数
            num_instances: 每类实例数
            aux_loss: 是否使用辅助损失
            use_temporal_attention: 是否使用时序注意力
            num_frames: 视频帧数
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.num_queries = (num_classes + 1) * num_instances
        self.Ns = 9
        self.transformer = transformer
        self.use_temporal_attention = use_temporal_attention
        self.num_frames = num_frames
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, backbone_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        self.background_prototype = nn.Parameter(torch.randn(backbone_dim) * 0.02)

        # Content queries
        self.content_proj = nn.Linear(backbone_dim, hidden_dim)
        self.instance_offsets = nn.Embedding(self.num_queries, hidden_dim)
        
        # Initialize anchor boxes - optimized for small objects in remote sensing
        def inverse_sigmoid(x, eps=1e-5):
            x = x.clamp(min=eps, max=1-eps)
            return torch.log(x/(1-x))

        # Initialize anchors with small object priors
        anchors = torch.rand(self.num_queries, 4)
        # Set smaller initial widths and heights for small objects (< 32x32 on 640x640 image)
        anchors[:, 2] = anchors[:, 2] * 0.3  # width
        anchors[:, 3] = anchors[:, 3] * 0.3  # height
        anchors = inverse_sigmoid(anchors)
        self.anchors = nn.Parameter(anchors)
        self.pos_query_mlp = MLP(4, hidden_dim, hidden_dim, 3)

        # Query Enhancement
        self.query_enhance_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.temperature = nn.Parameter(torch.FloatTensor([10.0]))

        # Temporal Attention for Video
        if use_temporal_attention:
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(hidden_dim)
            self.motion_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            )

        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        self.hidden_dim = hidden_dim


    def extract_backbone_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取图像特征和 query 特征
        
        Args:
            images: [B, 3, H, W]
        
        Returns:
            src: [B, C, H, W] backbone 特征图 (经过 FPN 融合)
        """
        # 转换为 NestedTensor
        if isinstance(images, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(images)
        else:
            samples = images
            
        # 通过 DETR 前向传播
        features, pos = self.backbone(samples)
        
        # Apply FPN for multi-scale feature fusion
        if hasattr(self.backbone, 'fpn'):
            fpn_features = self.backbone.fpn(features)
            # Use the highest resolution feature map (from layer1) for small objects
            src = fpn_features[0].tensors  # Highest resolution
        else:
            src, mask = features[-1].decompose()

        return src
    
    def compute_class_prototypes(self, support_images: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        计算类原型（ProtoNet 风格）
        
        Args:
            support_images: Dict mapping class_id to images [K, 3, H, W]
        
        Returns:
            prototypes: [num_classes, hidden_dim] 类原型向量
        """
        prototypes = []
        
        for class_id in sorted(support_images.keys()):
            images = support_images[class_id]  # [K, 3, H, W]
            K = images.size(0)
            
            # 提取特征 - 获取每个 support image 的 query 特征
            # 取所有 query 的平均作为图像级特征
            features = self.extract_backbone_features(images)  # [K, C, H, W]
            
            # 对每个 support image，取所有 query 的平均，然后对 K 个 shot 平均
            # [K, hidden_dim]
            image_features = features.mean(dim=[2,3])  # 平均所有 query

            # 对 K 个 shot 平均（ProtoNet 核心）
            prototype = image_features.mean(dim=0)  # [hidden_dim]
            prototypes.append(prototype)


        prototypes.append(self.background_prototype)
        prototypes = torch.stack(prototypes, dim=0)  # [num_classes, hidden_dim]
        
        return prototypes
    
    def sample_local_features(self, features: torch.Tensor, anchors: torch.Tensor, num_samples: int = 9) -> torch.Tensor:
        """Sample local features around anchor boxes using grid sampling"""
        B, C, H, W = features.shape
        num_queries = anchors.size(0)

        # Create sampling grid
        grid_size = int(math.sqrt(num_samples))
        
        sample_points = []
        for i in range(num_queries):
            cx, cy, w, h = anchors[i]
            
            x_coords = torch.linspace(cx - w/2, cx + w/2, grid_size, device=anchors.device)
            y_coords = torch.linspace(cy - h/2, cy + h/2, grid_size, device=anchors.device)
            
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            sample_points.append(grid_points)
        
        sample_points = torch.stack(sample_points, dim=0)
        sample_points = sample_points * 2 - 1  # Normalize to [-1, 1]
        
        # Sample features
        sample_points = sample_points.unsqueeze(0).expand(B, -1, -1, -1)
        sample_points = sample_points.reshape(B, num_queries * num_samples, 2)
        
        sampled_features = F.grid_sample(
            features, 
            sample_points.unsqueeze(2),
            align_corners=False,
            mode='bilinear'
        )  # [B, C, num_queries*num_samples, 1]
        
        sampled_features = sampled_features.squeeze(-1).permute(0, 2, 1)
        sampled_features = sampled_features.reshape(B, num_queries, num_samples, C)
        
        if B == 1:
            sampled_features = sampled_features.squeeze(0)
        
        return sampled_features

    def enhance_queries(self, content_queries: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        """Position-constrained class-aware query enhancement"""
        num_queries = content_queries.size(0)
        num_samples = local_features.size(1)
        
        # Flatten local features
        local_features_flat = local_features.reshape(-1, local_features.size(-1))
        
        # Cross-attention
        content_queries_expanded = content_queries.unsqueeze(1).expand(-1, num_samples, -1)
        content_queries_expanded = content_queries_expanded.reshape(-1, self.hidden_dim)
        
        attn_output, _ = self.query_enhance_attn(
            query=content_queries_expanded.unsqueeze(0),
            key=local_features_flat.unsqueeze(0),
            value=local_features_flat.unsqueeze(0)
        )
        
        attn_output = attn_output.squeeze(0).reshape(num_queries, num_samples, self.hidden_dim)
        
        # Compute similarity weights
        cos_sim = F.cosine_similarity(attn_output, content_queries.unsqueeze(1), dim=-1)
        weights = F.softmax(cos_sim * self.temperature, dim=-1)
        
        # Weighted aggregation
        enhanced_queries = (attn_output * weights.unsqueeze(-1)).sum(dim=1)
        
        return enhanced_queries

    def build_content_queries(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Build content queries from class prototypes"""
        content_queries = []
        for i in range(prototypes.size(0)):
            proto_expanded = prototypes[i:i+1].expand(self.num_instances, -1)
            queries = self.content_proj(proto_expanded) 
            content_queries.append(queries)
        
        content_queries = torch.cat(content_queries, dim=0)
        content_queries = (content_queries + self.instance_offsets.weight) * 0.5
        return content_queries

    def build_position_queries(self, anchors: torch.Tensor) -> torch.Tensor:
        """Build position queries from anchor boxes"""
        return self.pos_query_mlp(anchors)

    def forward_frame(self, frame: torch.Tensor, prototypes: torch.Tensor) -> Dict:
        """
        处理单帧图像
        
        Args:
            frame: [3, H, W] 单帧图像
            prototypes: [num_classes_with_bg, hidden_dim] 类原型（包含背景类）
        
        Returns:
            Dict 包含:
                - 'pred_boxes': [num_queries, 4] 预测的边界框
                - 'cls_scores': [num_queries, num_classes_with_bg] 分类分数（通过距离度量）
                    其中 num_classes_with_bg = num_foreground_classes + 1（背景类）
                - 'query_features': [num_queries, hidden_dim] query 特征
        """
        if isinstance(frame, (torch.Tensor)):
            frame = [frame]
        if isinstance(frame, (list, torch.Tensor)):
            frame = nested_tensor_from_tensor_list(frame)
        features, pos = self.backbone(frame)

        # Apply FPN for multi-scale feature fusion
        if hasattr(self.backbone, 'fpn'):
            fpn_features = self.backbone.fpn(features)
            src = fpn_features[0].tensors  # Use highest resolution for small objects
            mask = features[-1].mask
            # Generate position encoding for the fused feature map
            from util.misc import NestedTensor
            pos_enc = self.backbone[1](NestedTensor(src, mask))
        else:
            src, mask = features[-1].decompose()
            pos_enc = pos[-1]
        
        assert src is not None

        transofrmer_input = self.input_proj(src)
        content_queries = self.build_content_queries(prototypes)
        local_features = self.sample_local_features(transofrmer_input, self.anchors.sigmoid(), num_samples=self.Ns)
        enhanced_content = self.enhance_queries(content_queries, local_features)
        position_queries = self.build_position_queries(self.anchors.sigmoid())
        queries = position_queries + enhanced_content

        hs = self.transformer(transofrmer_input, mask, queries, pos_enc)[0]
        
        # 使用最后一层的输出
        # query_features: [B, num_queries, hidden_dim]
        query_features = self.class_embed(hs[-1])  # 现在输出的是特征而非 logits

        # 边界框预测
        pred_boxes = self.bbox_embed(hs[-1]).sigmoid()  # [B, num_queries, 4]
        
        # 确保是 2D 输出 [num_queries, 4]
        if pred_boxes.dim() == 3:
            pred_boxes = pred_boxes.squeeze(0)  # [num_queries, 4]
        if query_features.dim() == 3:
            query_features = query_features.squeeze(0)  # [num_queries, hidden_dim]
        
        # 计算欧氏距离的平方: [num_queries, num_classes]
        l2_dist_sq = pdist(query_features, prototypes)
        
        # 开方得到真正的欧氏距离，加 eps 防止梯度问题
        l2_dist = torch.sqrt(l2_dist_sq + 1e-8)  # [num_queries, num_classes]
        
        # 负距离作为相似度
        neg_dist = -l2_dist
        
        # 温度缩放: cls_scores = neg_dist * scale
        cls_scores = neg_dist * self.scale  # [num_queries, num_classes]
        
        return {
            'pred_boxes': pred_boxes,        # [num_queries, 4]
            'cls_scores': cls_scores,        # [num_queries, num_classes]
            'query_features': query_features,  # [num_queries, hidden_dim]
        }
    
    def forward(self, 
                frames: torch.Tensor, 
                support_images: Dict[int, torch.Tensor],
                return_all_frames: bool = True) -> List[Dict]:
        """
        前向传播（视频序列）
        
        Args:
            frames: [T, 3, H, W] 视频帧序列
            support_images: Dict mapping class_id to images [K, 3, H, W]
            return_all_frames: 是否返回所有帧的结果
        
        Returns:
            List[Dict]: 每个帧的检测结果，格式为 [{
                'pred_boxes': [num_queries, 4],
                'cls_scores': [num_queries, num_classes_with_bg],
                'query_features': [num_queries, hidden_dim],
            }, ...]
            其中 cls_scores 包含前景类和背景类（最后一维）
        """
        T = frames.size(0)
        
        # 计算类原型
        prototypes = self.compute_class_prototypes(support_images)  # [num_classes, hidden_dim]
        
        # 处理每一帧
        outputs = []
        for t in range(T):
            output = self.forward_frame(frames[t], prototypes)  # Dict
            outputs.append(output)
        
        if return_all_frames:
            return outputs
        else:
            return [outputs[-1]]
    
    def post_process(self, output, target, class_id_to_idx, confidence_threshold=0.5):
        cls_scores, boxes = output['cls_scores'], output['pred_boxes']
        img_h, img_w = target['orig_size']

        probs = F.softmax(cls_scores, -1)
        max_probs, pred_classes = probs.max(-1)
        
        keep = max_probs > confidence_threshold
        
        boxes_list = []
        scores_list = []
        labels_list = []
        
        idx_to_class_id = {v: k for k, v in class_id_to_idx.items()}
        
        for i in torch.where(keep)[0]:
            box = boxes[i].detach().cpu().numpy()
            cx, cy, w, h = box
            # Convert from [cx, cy, w, h] to [x1, y1, x2, y2] absolute coordinates
            x1 = (cx - 0.5 * w) * img_w
            y1 = (cy - 0.5 * h) * img_h
            x2 = (cx + 0.5 * w) * img_w
            y2 = (cy + 0.5 * h) * img_h
            boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
            scores_list.append(float(max_probs[i].item()))
            labels_list.append(idx_to_class_id.get(pred_classes[i].item(), pred_classes[i].item()))
        
        detections = {'boxes': boxes_list, 'scores': scores_list, 'labels': labels_list}


        gt_boxes_list = []
        gt_labels_list = []
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        for i in range(gt_boxes.shape[0]):
            gt_box = gt_boxes[i].detach().cpu().numpy()
            cx, cy, w, h = gt_box
            # Convert from [cx, cy, w, h] to [x1, y1, x2, y2] absolute coordinates
            x1 = (cx - 0.5 * w) * img_w
            y1 = (cy - 0.5 * h) * img_h
            x2 = (cx + 0.5 * w) * img_w
            y2 = (cy + 0.5 * h) * img_h
            gt_boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
            gt_labels_list.append(idx_to_class_id.get(gt_labels[i].item(), gt_labels[i].item()))

        annotations = {'boxes': gt_boxes_list, 'labels': gt_labels_list}

        return detections, annotations


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
           
           For small object detection:
           - Add scale-aware weights to give more importance to small objects
           - Normalize GIoU loss by object area
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # Scale-aware weighting for small objects
        # Calculate target box areas (w * h)
        target_areas = target_boxes[:, 2] * target_boxes[:, 3]
        
        # Define scale thresholds (normalized coordinates)
        # Small object: area < 0.01 (approximately 64x64 pixels on 640x640 image)
        # Tiny object: area < 0.0025 (approximately 32x32 pixels)
        scale_weights = torch.ones_like(target_areas)
        small_mask = target_areas < 0.01
        tiny_mask = target_areas < 0.0025
        
        # Give higher weights to smaller objects
        scale_weights[small_mask] = 2.0
        scale_weights[tiny_mask] = 4.0
        
        # Apply scale weights to bbox loss
        loss_bbox_weighted = loss_bbox * scale_weights.unsqueeze(1)
        
        losses = {}
        losses['loss_bbox'] = loss_bbox_weighted.sum() / num_boxes

        # Compute GIoU loss
        giou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        
        # Scale-normalized GIoU for small objects
        # Small objects have unstable GIoU, normalize by sqrt(area)
        scale_factor = torch.sqrt(target_areas + 1e-8)
        loss_giou = (1 - torch.diag(giou)) / scale_factor.clamp(min=0.05)
        
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_pc2aq(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    
    # 对于小样本检测，num_classes 应该是 n_way（前景类别数）
    # SetCriterion 内部会再 +1 来包含背景类
    num_classes = getattr(args, 'n_way', 5)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = PC2AQ(
        backbone,
        transformer,
        backbone_dim=args.dim_feedforward,
        num_classes=args.n_way,
        num_instances=16,
        aux_loss=args.aux_loss,
    )

    # 加载预训练权重（可选）
    if getattr(args, 'load_pretrained_detr', False):
        try:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", 
                map_location="cpu", 
                check_hash=True
            )
            state_dict = checkpoint["model"]
            
            # 获取模型当前的状态字典
            model_state = model.state_dict()
            
            # 统计信息
            total_keys = len(model_state)
            loaded_keys = 0
            missing_keys = []
            unexpected_keys = []
            
            # 只保留模型中存在的键（排除class_embed等自定义层）
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state:
                    if model_state[k].shape == v.shape:
                        filtered_state_dict[k] = v
                        loaded_keys += 1
                    else:
                        print(f"[Weight Load] Shape mismatch: {k} | checkpoint: {v.shape} vs model: {model_state[k].shape}")
                else:
                    unexpected_keys.append(k)
            
            # 检查模型中有哪些键没有被加载
            for k in model_state:
                if k not in filtered_state_dict:
                    missing_keys.append(k)
            
            # 加载权重
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # 打印验证结果
            print("=" * 60)
            print("Pretrained Weights Loading Summary:")
            print("=" * 60)
            print(f"Total model parameters: {total_keys}")
            print(f"Successfully loaded: {loaded_keys} ({loaded_keys/total_keys*100:.1f}%)")
            print(f"Missing (will be trained from scratch): {len(missing_keys)}")
            print(f"Ignored from checkpoint: {len(unexpected_keys)}")
            
            # 按模块统计
            backbone_loaded = sum(1 for k in filtered_state_dict if k.startswith('backbone.'))
            transformer_loaded = sum(1 for k in filtered_state_dict if k.startswith('transformer.'))
            backbone_total = sum(1 for k in model_state if k.startswith('backbone.'))
            transformer_total = sum(1 for k in model_state if k.startswith('transformer.'))
            
            print(f"\nModule-wise loading status:")
            print(f"  Backbone:   {backbone_loaded}/{backbone_total} loaded")
            print(f"  Transformer: {transformer_loaded}/{transformer_total} loaded")
            
            # 显示部分缺失的键（如果数量不多）
            if missing_keys and len(missing_keys) <= 10:
                print(f"\nMissing keys (newly initialized):")
                for k in missing_keys:
                    print(f"  - {k}")
            elif missing_keys:
                print(f"\nMissing keys (first 10 of {len(missing_keys)}):")
                for k in missing_keys[:10]:
                    print(f"  - {k}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            import traceback
            traceback.print_exc()


    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    device = torch.device(args.device if hasattr(args, 'device') else 'cuda')
    criterion.to(device)

    return model, criterion
