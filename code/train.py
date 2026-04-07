"""
FSDETR 训练脚本 - 小样本视频目标检测

基于 DETR 的训练流程，适配小样本视频目标检测任务
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.pc2aq import build_pc2aq
from dataloader import FSVODDataset, get_fsvod_loaders
import util.misc as utils

import copy
import contextlib
import io
from utils import calculate_map

def get_args_parser():
    parser = argparse.ArgumentParser('FSDETR training and evaluation', add_help=False)
    
    # 学习率参数
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Backbone 参数
    parser.add_argument('--load_pretrained_detr', default=True, type=bool,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer 参数
    parser.add_argument('--aux_loss', action='store_true',
                        help="Use auxiliary decoding losses (helpful for deeper models)")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss 相关参数
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help='Class coefficient in the matching cost')
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help='L1 box coefficient in the matching cost')
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help='giou box coefficient in the matching cost')
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help='Relative classification weight of the no-object class')

    # 小样本参数
    parser.add_argument('--n_way', default=5, type=int,
                        help='Number of classes per episode')
    parser.add_argument('--k_shot', default=5, type=int,
                        help='Number of support examples per class')
    parser.add_argument('--max_frames', default=16, type=int,
                        help='Maximum number of frames per video')

    # 数据集参数
    parser.add_argument('--data_root', default='dataset', type=str,
                        help='Root directory of the dataset')
    parser.add_argument('--img_size', default=560, type=int,
                        help='Image size')
    parser.add_argument('--episodes_per_epoch', default=100, type=int,
                        help='Number of episodes per epoch')
    parser.add_argument('--val_episodes', default=50, type=int,
                        help='Number of validation episodes')

    # 输出和日志参数
    parser.add_argument('--output_dir', default='outputs/fs_detr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency')

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    
    return parser


def prepare_targets(frame_annotations, classes, device):
    """
    将 frame_annotations 转换为 DETR 格式的 targets
    
    Args:
        frame_annotations: List[Dict] 每帧的标注，包含 'objects' 和 'original_size'
        classes: List[int] episode 中的类别列表
        device: 计算设备
    
    Returns:
        targets: List[Dict] 每帧的 target，包含:
            - 'boxes': Tensor[N, 4] - [cx, cy, w, h] 格式，归一化到 [0, 1]
            - 'labels': Tensor[N] - 0-based 类别索引
            - 'orig_size': Tensor[2] - 原始图像尺寸 [H, W]
            - 'size': Tensor[2] - 当前图像尺寸 [H, W]
    """
    # 构建类别到索引的映射
    class_to_idx = {cls_id: i for i, cls_id in enumerate(sorted(classes))}
    
    targets = []
    for frame_anno in frame_annotations:
        boxes = []
        labels = []
        
        # 获取原始图像尺寸 (H, W)
        orig_size = frame_anno.get('original_size', None)
        if orig_size is None:
            # 如果没有 original_size，使用默认值（应该从图像中读取）
            orig_h, orig_w = 560, 560  # 默认 target_size
        else:
            orig_h, orig_w = orig_size
        
        # 当前尺寸（经过 resize 后的尺寸）
        # 从 dataloader 中我们知道 target_size 是 (560, 560)
        curr_h, curr_w = 560, 560
        
        for obj in frame_anno['objects']:
            # bbox_scaled 是 [cx, cy, w, h] 格式，已经归一化到 [0, 1]（相对于 target_size）
            bbox = obj['bbox']
            # 处理可能的嵌套情况
            cx = bbox[0].item() if isinstance(bbox[0], torch.Tensor) else bbox[0]
            cy = bbox[1].item() if isinstance(bbox[1], torch.Tensor) else bbox[1]
            w = bbox[2].item() if isinstance(bbox[2], torch.Tensor) else bbox[2]
            h = bbox[3].item() if isinstance(bbox[3], torch.Tensor) else bbox[3]
            boxes.append([cx, cy, w, h])
            
            # 将类别 ID 映射到 0-based 索引
            cat_id = obj['category_id']
            if isinstance(cat_id, (list, tuple)):
                cat_id = cat_id[0]
            if cat_id in class_to_idx:
                labels.append(class_to_idx[cat_id])
        
        if len(boxes) > 0:
            targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32, device=device),
                'labels': torch.tensor(labels, dtype=torch.int64, device=device),
                'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.int64, device=device),
                'size': torch.tensor([curr_h, curr_w], dtype=torch.int64, device=device),
            })
        else:
            # 如果没有目标，添加空 target
            targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=device),
                'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.int64, device=device),
                'size': torch.tensor([curr_h, curr_w], dtype=torch.int64, device=device),
            })
    
    return targets


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, 
                    max_norm=0, print_freq=10):
    """
    训练一个 epoch
    
    对每个 episode：
    1. 计算 support prototypes
    2. 对 query frames 进行前向传播
    3. 计算每帧的损失
    """
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    
    for episode_data in metric_logger.log_every(data_loader, print_freq, header):
        # 提取数据
        support_images = {cls_id: imgs.to(device) for cls_id, imgs in episode_data['support_images'].items()}  # Dict[class_id -> Tensor[K, 3, H, W]]
        query_frames = episode_data['query_frames'].to(device)      # [T, 3, H, W]
        frame_annotations = episode_data['frame_annotations']
        classes = episode_data['classes']
        
        # 准备 targets（每帧一个 target）
        targets = prepare_targets(frame_annotations, classes, device)
        
        # 计算类原型
        prototypes = model.compute_class_prototypes(support_images)
        # prototypes: [num_classes + 1, hidden_dim] (包含背景类)
        
        # 处理每一帧
        total_loss = 0
        total_loss_dict = {}
        num_frames_with_targets = 0
        
        for t in range(query_frames.size(0)):
            # 单帧前向传播
            frame_output = model.forward_frame(query_frames[t], prototypes)
            # frame_output: {'pred_boxes': [num_queries, 4], 'cls_scores': [num_queries, num_classes+1], ...}
            
            # 跳过没有目标的帧
            if targets[t]['boxes'].shape[0] == 0:
                continue
            
            # 计算损失（注意：criterion 期望的输入格式）
            # 需要将输出转换为 criterion 期望的格式
            outputs_for_criterion = {
                'pred_logits': frame_output['cls_scores'].unsqueeze(0),  # [1, num_queries, num_classes+1]
                'pred_boxes': frame_output['pred_boxes'].unsqueeze(0),    # [1, num_queries, 4]
            }
            
            # 计算损失
            loss_dict = criterion(outputs_for_criterion, [targets[t]])
            
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            total_loss += losses
            
            # 累积损失统计
            for k, v in loss_dict.items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = 0
                total_loss_dict[k] += v.item()
            
            num_frames_with_targets += 1
        
        # 如果没有有效帧，记录一个虚拟损失避免 metric_logger 除零错误
        if num_frames_with_targets == 0:
            # 更新一个虚拟值，确保 metric_logger 有数据
            metric_logger.update(loss=0.0, loss_ce=0.0, loss_bbox=0.0, loss_giou=0.0)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            continue
        
        # 平均损失
        total_loss = total_loss / num_frames_with_targets
        
        # 检查损失是否有效
        loss_value = total_loss.item()
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, skipping this episode")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        # 更新日志
        avg_loss_dict = {k: v / num_frames_with_targets for k, v in total_loss_dict.items()}
        metric_logger.update(loss=loss_value, **avg_loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # 同步统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # 检查是否有数据，避免除零错误
    result = {}
    for k, meter in metric_logger.meters.items():
        if meter.count > 0:
            result[k] = meter.global_avg
        else:
            result[k] = 0.0  # 默认值
    return result


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, threshold=0.5):
    """
    验证模型，同时计算 mAP
    
    Args:
        model: FSDETR 模型
        criterion: 损失函数
        data_loader: 验证数据加载器
        device: 计算设备
        threshold: 检测置信度阈值
    
    Returns:
        Dict 包含损失和 mAP 指标
    """
    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    all_detections = []
    all_annotations = []
    for episode_data in metric_logger.log_every(data_loader, 10, header):
        # 提取数据
        support_images = {cls_id: imgs.to(device) for cls_id, imgs in episode_data['support_images'].items()}  # Dict[class_id -> Tensor[K, 3, H, W]]
        query_frames = episode_data['query_frames'].to(device)
        frame_annotations = episode_data['frame_annotations']
        classes = episode_data['classes']
        class_to_idx = {cls_id: i for i, cls_id in enumerate(sorted(classes))}

        # 准备 targets
        targets = prepare_targets(frame_annotations, classes, device)

        # 计算类原型
        prototypes = model.compute_class_prototypes(support_images)
        
        # 处理每一帧
        frame_predictions = []
        frame_targets = []
        
        for t in range(query_frames.size(0)):
            # 跳过没有目标的帧
            if targets[t]['boxes'].shape[0] == 0:
                continue

            # 单帧前向传播
            frame_output = model.forward_frame(query_frames[t], prototypes)

            # 计算损失
            outputs_for_criterion = {
                'pred_logits': frame_output['cls_scores'].unsqueeze(0),
                'pred_boxes': frame_output['pred_boxes'].unsqueeze(0),
            }
            
            loss_dict = criterion(outputs_for_criterion, [targets[t]])
            
            # 更新日志
            metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

            detections, annotations = model.post_process(frame_output, targets[t], class_to_idx)
            all_detections.append(detections)
            all_annotations.append(annotations)
    
    # Calculate mAP
    mAP = calculate_map(all_detections, all_annotations) if all_detections else 0.0
    print(f'mAP {mAP:.4f}')

    # 同步统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # 合并结果
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # 添加 mAP 到结果中
    results['mAP'] = mAP

    return results


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # 固定随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型
    model, criterion = build_pc2aq(args)
    model.to(device)
    
    # ============ 两阶段训练策略 ============
    # 阶段1：冻结 backbone 和 transformer，只训练新模块（anchors, pos_query_mlp, class_embed, bbox_embed）
    # 阶段2：解冻所有参数，端到端微调
    WARMUP_EPOCHS = 20  # 可根据需要调整
    
    new_module_keywords = ['anchors', 'pos_query_mlp', 'class_embed', 'bbox_embed', 
                           'background_prototype', 'scale', 'temperature']
    
    def freeze_pretrained_modules(m):
        """冻结预训练模块（backbone 和 transformer）"""
        for name, param in m.named_parameters():
            if 'backbone' in name or 'transformer' in name:
                param.requires_grad = False
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        print(f"Phase 1: Frozen backbone/transformer. Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
        
        # 打印哪些模块是可训练的
        print("Trainable modules:")
        for name, param in m.named_parameters():
            if param.requires_grad and any(kw in name for kw in new_module_keywords):
                print(f"  - {name}")
    
    def unfreeze_all_modules(m):
        """解冻所有模块"""
        for param in m.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        print(f"Phase 2: Unfrozen all modules. Trainable: {trainable:,}/{total:,} (100.0%)")
    
    # 初始阶段：冻结预训练模块
    is_warmup_phase = True
    freeze_pretrained_modules(model)
    
    # 构建优化器（只优化未冻结的参数）
    def build_optimizer(train_model, warmup=True):
        if warmup:
            # 阶段1：只优化新模块，使用较大学习率
            params = [p for p in train_model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            print(f"Optimizer: Phase 1 (warmup), lr={args.lr}")
        else:
            # 阶段2：分层学习率
            param_groups = [
                {
                    "params": [p for n, p in train_model.named_parameters() 
                              if any(kw in n for kw in new_module_keywords) and p.requires_grad],
                    "lr": args.lr,  # 新模块保持较大学习率
                },
                {
                    "params": [p for n, p in train_model.named_parameters() 
                              if not any(kw in n for kw in new_module_keywords) 
                              and "backbone" not in n and "transformer" not in n and p.requires_grad],
                    "lr": args.lr * 0.1,  # 其他模块中等学习率
                },
                {
                    "params": [p for n, p in train_model.named_parameters() 
                              if ("transformer" in n) and p.requires_grad],
                    "lr": args.lr_backbone,  # transformer 小学习率
                },
                {
                    "params": [p for n, p in train_model.named_parameters() 
                              if ("backbone" in n) and p.requires_grad],
                    "lr": args.lr_backbone * 0.1,  # backbone 极小学习率
                },
            ]
            # 过滤掉空的参数组
            param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
            opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            print(f"Optimizer: Phase 2 (full training), lr settings: new={args.lr}, other={args.lr*0.1}, transformer={args.lr_backbone}, backbone={args.lr_backbone*0.1}")
        return opt
    
    optimizer = build_optimizer(model, warmup=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)

    # 构建数据集
    print("Loading data...")
    data_loaders = get_fsvod_loaders(
        root_dir=args.data_root,
        n_way=args.n_way,
        k_shot=args.k_shot,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.img_size, args.img_size)
    )
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']

    output_dir = Path(args.output_dir)
    
    # 恢复训练
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # 评估模式
    if args.eval:
        test_stats = evaluate(model, criterion, test_loader, device)
        print(test_stats)
        return

    # 训练循环
    print("Start training")
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(args.start_epoch, args.epochs):
        # ========== 阶段切换：从 warmup 切换到完整训练 ==========
        if is_warmup_phase and epoch == WARMUP_EPOCHS:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}: Switching from Phase 1 to Phase 2")
            print(f"Unfreezing backbone and transformer...")
            print(f"{'='*60}\n")
            
            is_warmup_phase = False
            unfreeze_all_modules(model)
            
            # 重新创建优化器（包含所有参数的分层学习率）
            optimizer = build_optimizer(model, warmup=False)
            # 重置学习率调度器
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[50], gamma=0.1)
        
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            args.clip_max_norm, args.print_freq
        )
        lr_scheduler.step()
        
        # 初始化日志记录
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        # 验证
        if (epoch + 1) % 5 == 0:
            val_stats = evaluate(model, criterion, test_loader, device)
            
            # 添加验证指标到日志
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})
            
            # 保存最佳模型
            val_loss = val_stats.get('loss', float('inf'))
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = output_dir / 'best_checkpoint.pth'
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_stats': val_stats,
                }, checkpoint_path)
                print(f"*** Saved best model at epoch {epoch} ***")
        
        # 定期保存检查点
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 20 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 记录日志
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FSDETR training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
