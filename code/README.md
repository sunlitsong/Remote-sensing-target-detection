# 项目交接文档 - PC2AQ: Position-Constrained Class-Aware Queries

## 项目概述

本项目是基于 DETR 的小样本视频目标检测 (Few-Shot Video Object Detection, FSVOD) 实现。

---

## 项目结构

```
.
├── models/
│   ├── pc2aq.py              # 主模型 (最终版本)
│   ├── backbone.py           # ResNet Backbone + Position Encoding
│   ├── transformer.py        # DETR Transformer
│   ├── matcher.py            # 匈牙利匹配器
│   └── position_encoding.py  # 位置编码
├── util/
│   ├── misc.py               # 工具函数 (NestedTensor等)
│   ├── box_ops.py            # 边界框操作
│   └── plot_utils.py         # 可视化工具
├── dataloader.py             # FSVOD 数据集加载器
├── train.py                  # 训练脚本
├── utils.py                  # mAP 计算等辅助函数
├── config.py                 # 配置参数
└── requirement.txt           # 依赖包
```

---

## 核心模型架构 (PC2AQ)

### 1. 整体流程

```
Support Images (N-way K-shot)
    ↓
Backbone Feature Extraction
    ↓
Class Prototypes (ProtoNet风格) [num_classes, hidden_dim]
    ↓
Query Video Frames [T, 3, H, W]
    ↓
For each frame:
    - Backbone → Feature Maps
    - Build Content Queries (from prototypes)
    - Build Position Queries (from anchors)
    - Query Enhancement (Local Feature Sampling + Cross-Attention)
    - Transformer Decoder
    - Classification (Distance-based) + Bounding Box Regression
```

### 2. 关键组件

#### 2.1 类别原型计算 (`compute_class_prototypes`)

```python
# 对每个类别的 K 个 support images:
features = backbone(images)  # [K, C, H, W]
image_features = features.mean(dim=[2,3])  # [K, C]
prototype = image_features.mean(dim=0)     # [C] - ProtoNet核心
```

#### 2.2 查询构建

- **Content Queries**: 从类别原型投影得到 `[num_classes+1, num_instances, hidden_dim]`
  ```python
  content_queries = content_proj(prototypes) + instance_offsets
  ```

- **Position Queries**: 从可学习锚框投影得到 `[num_queries, hidden_dim]`
  ```python
  position_queries = pos_query_mlp(anchors.sigmoid())
  ```

#### 2.3 查询增强 (`query_enhance_attn`)

```python
# 1. 在锚框位置采样局部特征 (9个网格点)
local_features = sample_local_features(features, anchors, num_samples=9)

# 2. Cross-Attention: Content Queries ↔ Local Features
enhanced_queries = cross_attention(content_queries, local_features)

# 3. 余弦相似度加权
weights = softmax(cosine_similarity * temperature, dim=-1)
enhanced_queries = (attn_output * weights.unsqueeze(-1)).sum(dim=1)
```

#### 2.4 分类策略 (Distance-based)

使用欧氏距离替代传统的分类头：

```python
# L2距离计算
l2_dist = sqrt(pdist(query_features, prototypes) + 1e-8)

# 负距离作为logits，温度缩放
cls_scores = -l2_dist * scale  # [num_queries, num_classes+1]
```

---

## 训练流程

### 阶段1: Warmup (前20 epochs)

```python
# 冻结 backbone 和 transformer
冻结: backbone.*, transformer.*
训练: anchors, pos_query_mlp, class_embed, bbox_embed, 
      background_prototype, scale, temperature, instance_offsets
      
优化器: AdamW, lr=1e-4
```

### 阶段2: 全参数微调 (20 epochs以后)

```python
# 解冻所有参数
分层学习率:
- 新模块 (anchors, class_embed等): lr=1e-4
- 其他模块: lr=1e-5
- Transformer: lr=1e-5
- Backbone: lr=1e-6

优化器: AdamW + MultiStepLR(milestones=[50], gamma=0.1)
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 1e-4 | 基础学习率 |
| `--lr_backbone` | 1e-5 | Backbone学习率 |
| `--batch_size` | 1 | Episode批次大小 |
| `--epochs` | 100 | 总训练轮数 |
| `--n_way` | 5 | 每episode类别数 |
| `--k_shot` | 5 | Support样本数 |
| `--max_frames` | 16 | 每视频最大帧数 |
| `--img_size` | 560 | 图像尺寸 |

---

## 损失函数

使用 DETR 标准损失组合：

```python
loss = λ_ce * loss_ce + λ_bbox * loss_bbox + λ_giou * loss_giou

其中:
- loss_ce: 交叉熵分类损失
- loss_bbox: L1边界框回归损失
- loss_giou: GIoU边界框损失

默认权重:
- λ_ce = 1
- λ_bbox = 5
- λ_giou = 2
```

---

## 数据集格式

### 目录结构

```
dataset/
├── train_annotations.json
├── val_annotations.json
├── test_annotations.json
└── videos/
    ├── video_001/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── video_002/
        └── ...
```

### 标注格式 (JSON)

```json
{
    "video_001": {
        "frames": {
            "000001": {
                "objects": [
                    {
                        "category_id": 1,
                        "bbox": [x, y, w, h]  // 归一化坐标
                    }
                ]
            }
        }
    }
}
```

---

## 运行命令

### 训练

```bash
python train.py \
    --data_root ./dataset \
    --output_dir ./outputs \
    --n_way 5 \
    --k_shot 5 \
    --max_frames 16 \
    --epochs 100 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --batch_size 1 \
    --num_workers 4
```

### 评估

```bash
python train.py \
    --data_root ./dataset \
    --resume ./outputs/best_checkpoint.pth \
    --eval
```

---

## 关键超参数

### 模型架构

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_dim` | 256 | Transformer隐藏维度 |
| `num_queries` | (n_way+1) * 16 | 查询数量 |
| `num_instances` | 16 | 每类别实例数 |
| `Ns` | 9 | 局部采样点数 (3x3网格) |
| `backbone` | resnet50 | 主干网络 |

### 关键可学习参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `anchors` | [num_queries, 4] | 可学习锚框 (cx, cy, w, h) in logit space |
| `instance_offsets` | Embedding | 实例级别偏移 |
| `background_prototype` | [hidden_dim] | 背景类原型 |
| `scale` | Scalar | 距离缩放因子 |
| `temperature` | Scalar | 注意力温度 |

---

## 预训练权重加载

支持从官方 DETR 加载预训练权重：

```python
# 自动从URL加载
checkpoint = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
)

# 只加载匹配的层 (backbone + transformer)
# 新模块 (class_embed, bbox_embed等) 随机初始化
```

在 `train.py` 中设置 `--load_pretrained_detr True` 启用（默认开启）。

---

## 注意事项

1. **batch_size**: 由于是小样本episode训练，batch_size通常设为1
2. **内存占用**: 每帧独立前向传播，视频帧数多时显存需求大
3. **类别索引**: 内部使用0-based索引，背景类索引为 `n_way`
4. **坐标格式**: 内部使用 [cx, cy, w, h] 归一化到 [0, 1]
