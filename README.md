# PC2AQ: Position-Constrained Class-Aware Queries

基于 DETR 的小样本视频目标检测 (Few-Shot Video Object Detection, FSVOD) 实现。

## 项目结构

```
/workspace
├── code/
│   ├── models/
│   │   ├── pc2aq.py              # 主模型 (最终版本)
│   │   ├── backbone.py           # ResNet Backbone + Position Encoding
│   │   ├── transformer.py        # DETR Transformer
│   │   ├── matcher.py            # 匈牙利匹配器
│   │   └── position_encoding.py  # 位置编码
│   ├── util/
│   │   ├── misc.py               # 工具函数 (NestedTensor等)
│   │   ├── box_ops.py            # 边界框操作
│   │   └── plot_utils.py         # 可视化工具
│   ├── dataloader.py             # FSVOD 数据集加载器
│   ├── train.py                  # 训练脚本
│   ├── utils.py                  # mAP 计算等辅助函数
│   ├── config.py                 # 配置参数
│   ├── requirement.txt           # 依赖包
│   └── README.md                 # 详细文档
└── README.md                     # 本文件
```

## 快速开始

### 环境安装

```bash
cd code
pip install -r requirement.txt
```

### 训练

```bash
cd code
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
cd code
python train.py \
    --data_root ./dataset \
    --resume ./outputs/best_checkpoint.pth \
    --eval
