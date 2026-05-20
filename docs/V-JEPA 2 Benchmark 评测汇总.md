# V-JEPA 2 Benchmark 评测汇总

> 本文档汇总了 V-JEPA 2 / V-JEPA 2.1 项目中支持的所有下游评测 Benchmark，并详细展开 **Something-Something V2 (SSv2)** 的视频分类评测流程。

---

## 零、V-JEPA 与 ViT 的关系

**V-JEPA**（Video Joint Embedding Predictive Architecture）是一个**自监督学习框架/方法**，而不是具体的神经网络架构。

**ViT**（Vision Transformer）是 V-JEPA 使用的**骨干网络（Backbone / Encoder）**。

### 训练阶段

V-JEPA 2 预训练时包含三个核心组件：

| 组件 | 类型 | 作用 |
|------|------|------|
| `encoder` | ViT | 在线编码器，处理输入视频的可见 patch |
| `target_encoder` | ViT (EMA) | 目标编码器，encoder 的指数移动平均副本，提供预测目标 |
| `predictor` | Transformer | 预测器，在 latent space 中预测被 mask 掉的 patch 的特征表示 |

**训练目标**：给定一段视频的可见 patch tokens，通过 `encoder` + `predictor` 预测被 mask 掉的 patch 在 `target_encoder` 输出空间中的特征表示。

### 评测阶段

所有下游 Benchmark 评测的都是 **V-JEPA 预训练好的 `target_encoder`（即 ViT）** 的表征能力：

1. 从 V-JEPA checkpoint 中加载 `target_encoder` 的权重
2. 构建 ViT 模型，将权重加载进去
3. **冻结 ViT**（`requires_grad = False`，不更新 backbone）
4. 在 ViT 输出的 patch tokens 上训练一个轻量的 **Attentive Classifier（Probe）**
5. 用 Probe 在下游任务上的性能来评估 V-JEPA 预训练的质量

> **一句话总结**：V-JEPA 是"训练方法"，ViT 是"模型架构"；我们评测的是 V-JEPA 训练出来的 ViT 在下游任务上强不强。

---

## 一、Benchmark 总览

### 1. 视频分类（Video Classification）— Frozen Probes

在冻结的 V-JEPA 2 backbone 上训练 Attentive Probe 进行视频分类评测。

| 简称 | 全称 | 类别数 | 任务类型 | 代码路径 | 配置路径 |
|------|------|--------|----------|----------|----------|
| **SSv2** | Something-Something V2 | 174 | 动作识别 | `evals/video_classification_frozen/` | `configs/eval/vitl/ssv2.yaml` |
| **K400** | Kinetics-400 | 400 | 动作识别 | 同上 | `configs/eval/vitl/k400.yaml` |
| **Diving48** | Diving48 | 48 | 细粒度动作分类 | 同上 | `configs/eval/vitl/diving48.yaml` |
| **COIN** | COIN | — | 指令性视频分类 | 同上 | `configs/eval/vitl/coin.yaml` |
| **Jester** | Jester | — | 手势识别 | 同上 | `configs/eval/vitl/jester.yaml` |

### 2. 图像分类（Image Classification）— Frozen Probes

| 简称 | 全称 | 类别数 | 任务类型 | 代码路径 | 配置路径 |
|------|------|--------|----------|----------|----------|
| **IN1K** | ImageNet-1K | 1,000 | 图像分类 | `evals/image_classification_frozen/` | `configs/eval/vitl/in1k.yaml` |

### 3. 动作预测（Action Anticipation）— Frozen Probes

| 简称 | 全称 | 任务类型 | 代码路径 | 配置路径 |
|------|------|----------|----------|----------|
| **EK100** | EPIC-KITCHENS-100 | 动作预测 | `evals/action_anticipation_frozen/` | `configs/eval/vitl/ek100.yaml` |
| **COIN_anticipation** | COIN Anticipation | 动作预测 | 同上 | — |

### 4. 视频问答（Video QA）— 论文报告

| Benchmark | 任务类型 | 备注 |
|-----------|----------|------|
| **MVP** | Video QA | 论文报告结果 44.5%，**代码库未开源评测代码** |
| **TempCompass** | Video QA | 论文报告结果 76.9%，**代码库未开源评测代码** |

### 5. 机器人操作（Robot Manipulation）— V-JEPA 2-AC

使用 Franka 机械臂评测，输入为单目 RGB 相机。

| 任务 | 子任务 | 代码路径 |
|------|--------|----------|
| **Reach** | 到达目标 | `app/vjepa_droid/` |
| **Grasp** | Cup / Box 抓取 | 同上 |
| **Pick-and-Place** | Cup / Box 搬运 | 同上 |

---

## 二、SSv2 评测流程详解

### 2.1 评测概述

**Something-Something V2 (SSv2)** 是一个大规模视频动作识别数据集，专注于**以动作为中心**的识别任务（例如 "把东西放进某物"、"从某物中取出东西"）。

V-JEPA 2 对 SSv2 的评测方式是 **Frozen Probe**：

1. **加载 V-JEPA 预训练的 ViT encoder**（从 `vitl.pt` 中提取 `target_encoder` 权重）
2. **冻结 ViT backbone**（`requires_grad = False`，整个评测过程不更新 backbone）
3. 在 ViT 输出的 patch tokens 上训练一个轻量的 **Attentive Classifier（Probe）**
4. 用 Probe 的分类准确率来评估 **V-JEPA 自监督预训练的质量**

> 所以 SSv2 的 Top-1 准确率（73.7% / 77.3%）反映的正是 **V-JEPA 训练出来的 ViT 特征在动作识别任务上的表征能力**。

### 2.2 模型配置

#### V-JEPA Checkpoint 加载

SSv2 评测加载的是 **V-JEPA 2 预训练好的 `target_encoder`**，而不是从头训练的 ViT。加载逻辑在 `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` 中：

```python
checkpoint = torch.load("vitl.pt", map_location="cpu")
# 从 V-JEPA checkpoint 中提取 target_encoder 的权重
pretrained_dict = checkpoint["target_encoder"]
# 构建 ViT-Large 模型
model = vit_large(img_size=256, num_frames=16, ...)
# 加载权重
model.load_state_dict(pretrained_dict, strict=False)
# 冻结 backbone
model.eval()
for p in model.parameters():
    p.requires_grad = False
```

#### 模型参数

| 配置项 | 参数 | 说明 |
|--------|------|------|
| Backbone | `vit_large` (ViT-L/16) | V-JEPA 的 target_encoder 就是 ViT-L |
| Patch Size | 16 | 空间 patch 大小 |
| Tubelet Size | 2 | 时间 tubelet 大小（每 2 帧一个 temporal token） |
| 预训练权重 | `vitl.pt` | V-JEPA 2 自监督预训练 checkpoint |
| 权重加载 Key | `target_encoder` | 从 checkpoint 中读取 target_encoder 的 state_dict |
| 位置编码 | RoPE (`use_rope: true`) | 旋转位置编码 |
| 最大帧数 | 128 | wrapper 支持的最大帧数 |

**Attentive Classifier (Probe)：**

| 配置项 | 参数 |
|--------|------|
| 类型 | `AttentiveClassifier` |
| Attention Heads | 16 |
| Probe Blocks | 4 |
| 输入 | Backbone 输出的 patch tokens |

### 2.3 数据配置

| 配置项 | 参数 | 说明 |
|--------|------|------|
| Dataset Type | `VideoDataset` | 使用 decord 加载视频 |
| 类别数 | 174 | SSv2 的动作类别 |
| 每片段帧数 | 16 | `frames_per_clip: 16` |
| 帧步长 | 4 | `frame_step: 4` |
| 分辨率 | 256 | `resolution: 256` |
| 训练数据 | `ssv2_train_paths.csv` | 视频路径列表 |
| 验证数据 | `ssv2_val_paths.csv` | 视频路径列表 |

**多片段测试（Multi-Clip Testing）：**

| 配置项 | 参数 | 说明 |
|--------|------|------|
| num_segments | 2 | 每段视频采样 2 个 temporal clips |
| num_views_per_segment | 3 | 每个 clip 采样 3 个 spatial crops |
| 总视图数 | 2 × 3 = **6** | 最终对 6 个视图的结果取平均 |

### 2.4 训练配置（Multi-Head Probe）

SSv2 评测使用 **Multi-Head** 训练策略：同时训练多个 probe head，每个 head 使用不同的超参数，最终选择表现最好的 head。

**训练参数：**

| 配置项 | 参数 |
|--------|------|
| Epochs | 20 |
| Batch Size | 4 (per GPU) |
| 数据类型 | `bfloat16` |
| 优化器 | AdamW (推测) |
| 学习率调度 | Cosine Annealing |

**Multi-Head 超参数组合（共 20 个 heads）：**

| Head 组 | Weight Decay | 学习率 (LR) |
|---------|--------------|-------------|
| 1-5 | 0.01 | 0.005, 0.003, 0.001, 0.0003, 0.0001 |
| 6-10 | 0.1 | 0.005, 0.003, 0.001, 0.0003, 0.0001 |
| 11-15 | 0.4 | 0.005, 0.003, 0.001, 0.0003, 0.0001 |
| 16-20 | 0.8 | 0.005, 0.003, 0.001, 0.0003, 0.0001 |

> 每个 head 独立训练，互不干扰。最终选取验证集上 Top-1 准确率最高的 head 作为最终结果。

### 2.5 推理配置

推理配置文件位于 `configs/inference/vitl/ssv2.yaml`，与训练配置的区别在于：

| 配置项 | 训练 | 推理 |
|--------|------|------|
| `val_only` | `false` | `true` |
| `lr` | 多个非零值 | `0.0`（冻结 probe 权重） |
| `weight_decay` | 多个非零值 | `0.0` |

推理流程：
1. 加载训练好的 backbone 和 probe checkpoint
2. 对验证集视频进行 multi-clip 推理（2 segments × 3 views）
3. 对 6 个视图的 logits 取平均
4. 输出 Top-1 / Top-5 准确率

### 2.6 运行命令

**训练 Probe（本地多 GPU）：**

```bash
python -m evals.main \
  --fname configs/eval/vitl/ssv2.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

**训练 Probe（SLURM 分布式）：**

```bash
python -m evals.main_distributed \
  --fname configs/eval/vitl/ssv2.yaml \
  --time 8600 --account your_account --qos your_qos
```

**推理（加载已训练 probe）：**

```bash
python -m evals.main \
  --fname configs/inference/vitl/ssv2.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
```

### 2.7 官方结果

| 模型 | 分辨率 | 帧数 | Multi-Clip | Top-1 |
|------|--------|------|------------|-------|
| ViT-L/16 | 256 | 16 | 2×3 | 73.7% |
| ViT-g/16 | 384 | 16 | 64×2×3 | **77.3%** |

> ViT-g/16_384 结果使用了更多的 temporal segments（64 segments），其余配置类似。

### 2.8 关键代码文件

| 文件 | 作用 |
|------|------|
| `evals/video_classification_frozen/eval.py` | 主评测循环（训练 + 推理），调用 `init_module` 加载 V-JEPA 权重 |
| `evals/video_classification_frozen/models.py` | 模型初始化入口，负责构建冻结的 encoder 和可训练的 classifiers |
| `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | **核心：加载 V-JEPA checkpoint，提取 `target_encoder` 权重到 ViT，并包装为 ClipAggregation** |
| `src/models/vision_transformer.py` | ViT 架构实现（`vit_large`, `vit_huge`, `vit_giant` 等） |
| `src/models/attentive_pooler.py` | AttentiveClassifier（Probe）实现 |
| `src/datasets/video_dataset.py` | VideoDataset 数据加载 |
| `configs/eval/vitl/ssv2.yaml` | 训练配置 |
| `configs/inference/vitl/ssv2.yaml` | 推理配置 |

---

## 三、备注

1. **V-JEPA 2.1** 使用相同的评测框架，但 backbone 换成了 2.1 版本预训练的权重（支持 ViT-B/L/g/G）。
2. **Video QA** 类 benchmark（MVP、TempCompass）在论文中有报告结果，但当前代码库 `evals/` 目录下没有对应的评测实现，可能需要外部工具链。
3. **Robot Manipulation** 评测属于 V-JEPA 2-AC，代码在 `app/vjepa_droid/` 下，使用 DROID 机器人轨迹数据进行后训练。
