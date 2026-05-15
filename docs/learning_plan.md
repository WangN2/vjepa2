# V-JEPA 2 快速学习计划

> 学习目标：从零 DL 基础到能理解、运行、修改 V-JEPA 2 代码
> 预计时间：10-14 天（每天 2-4 小时）

---

## 阶段一：补 DL 基础（Day 1-3）

### 核心概念速查

| 概念 | 重要程度 | 推荐资源 |
|------|---------|---------|
| 自监督学习 vs 监督学习 vs 无监督学习 | ★★★★★ | [LeCun 自监督学习科普](https://www.bilibili.com/video/BV15txgzNEGu/?spm_id_from=333.337.search-card.all.click&vd_source=6531904af9bee88da85d6f4a50423ba7)（25 分钟） |
| Vision Transformer (ViT) 原理 | ★★★★★ | 李沐 [ViT 讲解](https://www.youtube.com/watch?v=6YF2VGrK8Jk)（1.5x 速，30 分钟） |
| EMA / 动量编码器 | ★★★★☆ | MoCo 论文示意图 — 理解 why > how |
| RoPE 位置编码 | ★★★☆☆ | RoFormer 论文的 2D 旋转部分 |
| L1 Loss vs L2 Loss | ★★★☆☆ | 知道 L1 对异常值更鲁棒即可 |
| 表征学习（Representation Learning）| ★★★★★ | 核心问题：模型学到了什么「有用的表示」，而不是死记硬背 |

### 重点理解

- **自监督学习**：没有标签，让模型从数据自身结构中学习。V-JEPA 的方式是「预测被遮罩的部分」
- **表征学习**：模型的目标不是分类/检测，而是学到**通用的视频特征**，可以迁移到各种下游任务
- **ViT 核心**：图片→16×16 小块（patches）→展平→加位置编码→Transformer→输出特征

---

## 阶段二：跑通整个流程（Day 4-5）

### Step 1：环境搭建

```bash
conda create -n vjepa2 python=3.12
conda activate vjepa2
pip install -e .
```

macOS 注意：decord 可能装不上，不影响代码阅读。

### Step 2：运行 demo

```bash
# 下载预训练模型
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt

# 运行 demo
# notebooks/vjepa2_demo.ipynb
```

在 Jupyter 里逐 cell 执行，每步加 print 看张量形状。

### Step 3：张量形状追踪

理解前向传播的数据流是**最重要的练习**：

```
输入视频: (1, 3, 16, 224, 224)
  ↓ PatchEmbed3D (patch_size=16, tubelet_size=2)
  特征: (1, 1568, 1408)
  1568 = (16/2) * (224/16) * (224/16) = 8 * 14 * 14 个 tokens
  ↓ Encoder
  编码后: (1, 1568, 1408)
  ↓ Predictor
  预测结果: (1, npred_tokens, pred_dim)
```

关键代码位置：
- [src/models/vision_transformer.py] → `PatchEmbed3D`
- [src/models/predictor.py] → `Predictor.forward`
- [app/vjepa/train.py] → `loss_fn`

### Step 4：训练循环数据流

读 `app/vjepa/train.py` 的 `train_one_epoch`，画图：

```
video → 随机遮罩(masks_enc, masks_pred)
      → encoder(video, masks_enc) → 可见 token 表示
      → predictor(可见 token, mask_tokens) → 预测被遮罩
      → target_encoder(video, masks_pred) → 目标特征
      → loss_fn(预测, 目标) → L1 loss
      → backward + optimizer + EMA
```

---

## 阶段三：深入代码（Day 6-9）

### Day 6：Vision Transformer

逐行读 [src/models/vision_transformer.py]：
- `PatchEmbed3D` — 3D 卷积如何把视频切成 patches
- `Block` — Transformer block 组成
- `RoPEAttention` / `Attention` — 两种注意力
- `VisionTransformer.forward` — 完整前向

练习：修改 `tubelet_size`，观察形状变化。

### Day 7：遮罩策略

读 [src/masks/multiseq_multiblock3d.py]：
- `_MaskGenerator.__call__` — 遮罩块生成逻辑
- 配置参数如何控制遮罩行为

练习：写脚本生成遮罩并可视化。

### Day 8：训练循环对比

对比 [app/vjepa/train.py] vs [app/vjepa_2_1/train.py]：
- V2.1 新增的 `compute_mask_distance` + 上下文损失
- 多级 hierarchical feature 拼接
- `predict_all` 模式

### Day 9：Transformer 内部组件

读 [src/models/utils/modules.py]：
- `SwiGLUFFN` — 比标准 FFN 多了什么
- `apply_rotary_emb_xyz` — 3D RoPE 实现

---

## 阶段四：动手实验（可选，有 GPU 时）

1. 用 `configs/train_2_1/vitb16/` 训练小模型
2. 修改损失函数，V2 的 loss_fn 中尝试 L2
3. 简化遮罩策略，从 2 层改为 1 层

---

## 学习资源速查表

| 你想理解什么 | 看哪个文件 | 关键类/函数 |
|------------|-----------|------------|
| 视频如何变成 tokens | [src/models/utils/patch_embed.py] | `PatchEmbed3D` |
| Encoder 结构 | [src/models/vision_transformer.py] | `VisionTransformer` |
| 预测器原理 | [src/models/predictor.py] | `Predictor.forward` |
| 遮罩如何生成 | [src/masks/multiseq_multiblock3d.py] | `_MaskGenerator.__call__` |
| 损失函数 | [app/vjepa/train.py] | `loss_fn` |
| EMA 更新 | [app/vjepa/train.py] | EMA 相关代码 |
| 多数据联合训练 | [src/utils/wrappers.py] | `MultiSeqWrapper` |
| 学习率调度 | [src/utils/schedulers.py] | `WarmupCosineSchedule` |
| Attentive Probe | [src/models/attentive_pooler.py] | `AttentivePooler` |
| 评估循环 | [evals/video_classification_frozen/eval.py] | `train_one_epoch` |
| 分布式训练 | [src/utils/distributed.py] | `init_distributed` |
| V2 vs V2.1 差异 | 对比两个 train.py | 上下文损失、hierarchical 特征 |
