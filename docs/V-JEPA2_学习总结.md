# V-JEPA 2 学习总结

## 一、项目概览

**V-JEPA 2**（Video Joint Embedding Predictive Architecture 2）是 Meta FAIR 团队提出的**自监督视频表征学习框架**。核心理念：让模型从大量无标注视频中，通过**在隐空间中预测被遮挡的视频区域**来学习物理世界的运动、交互和时序理解能力。

项目包含三个主要版本：

| 版本 | 发布时间 | 核心贡献 | 最大模型 |
|------|----------|----------|----------|
| V-JEPA 2 | 2025-06 | 自监督视频预训练 + 机器人后训练 | ViT-g/16 (1B) |
| V-JEPA 2-AC | 2025-06 | 动作条件世界模型（机器人操控） | ViT-g/16 (1B) |
| V-JEPA 2.1 | 2026-03 | 稠密特征学习改进（新训练配方） | ViT-G/16 (2B) |

论文引用：
```bibtex
@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}

@article{murlabadia2026vjepa2_1,
  title={V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning},
  author={Mur-Labadia, Lorenzo and Muckley, Matthew and Bar, Amir and Assran, Mahmoud and
Sinha, Koustuv and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas and Bardes, Adrien},
  journal={arXiv preprint arXiv:2603.14482},
  year={2026}
}
```

---

## 二、核心思想（一句话）

> **用视频教模型「预测未来」：给模型看视频的一部分，让它预测被遮住的部分在「特征空间」里应该长什么样。**

这和 MAE（Masked Autoencoder）很像，但 MAE 是在像素空间做预测，而 V-JEPA 在**语义特征空间**做预测——这就像让学生预测概念而不是抄写词语，学到的表征更抽象、更通用。

---

## 三、模型架构详解

### 3.1 整体框架（三件套）

```
                         Training Pipeline
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  视频 ───► Encoder (online) ───► Predictor ───► 预测被遮罩的特征  │
│             │                          │                        │
│             │ EMA 更新                 │ L1 Loss                │
│             ▼                          ▼                        │
│  视频 ───► Target Encoder (EMA) ───► 目标特征（Ground Truth）    │
│             (freeze, no grad)                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### (a) Encoder（编码器）—— 理解视频的骨干网络

**文件：** `src/models/vision_transformer.py`

基于 Vision Transformer（ViT），支持多种规模：

| 模型变体 | 参数量 | embed_dim | depth | num_heads | 备注 |
|---------|--------|-----------|-------|-----------|------|
| ViT-L | 300M | 1024 | 24 | 16 | V2 最小版本 |
| ViT-H | 600M | 1280 | 32 | 16 | |
| ViT-g | 1B | 1408 | 40 | 22 (xformers) | 主力模型 |
| ViT-G | 2B | 1664 | 48 | 26 (xformers) | V2.1 新增 |

**关键设计：**

- **PatchEmbed3D**：使用 3D 卷积将视频切分成 spatio-temporal patches。每帧切成 16×16 的 patch，tubelet_size=2 意味着每 2 帧合并为一个时间步。输入 16 帧 → 8 个时间步。

- **位置编码**：支持两种方案：
  - 3D sincos 位置编码（固定/不可学习）
  - **RoPE（旋转位置编码）**——将 head_dim 分成 3 份（depth/height/width 各占 1/3），分别做旋转编码，能自然地处理变长/变分辨率输入。代码实现在 `src/models/utils/modules.py` 的 `RoPEAttention` 类中。

- **SwiGLU FFN**：可选的 FFN 结构，`SwiGLUFFN` 实现。通过 `wide_silu` 控制宽度。

- **Deep Self-Supervision（V2.1 新增）**：从 4 个层级提取特征（如 depth=40 时取 layers [9,19,29,39]），拼接后传给 predictor，让模型学到层次化的稠密特征。

- **Modality Embedding（V2.1 新增）**：可学习的 `img_mod_embed` / `video_mod_embed`，让同一个 encoder 同时处理图片和视频。

- **`MultiSeqWrapper`**：包装 encoder 以支持多数据集、不同帧率的批量输入。核心是为每个数据集（不同 fpc）分别处理，然后在 batch 维度拼接。

#### (b) Predictor（预测器）—— 在特征空间做推理

**文件：** `src/models/predictor.py`

一个独立的较小 Transformer，作用是从 encoder 输出的「可见 token」**预测「被遮罩 token」** 的目标特征。

**流程：**
```
可见 token ──► Linear投影到预测维度 ──┐
                                      ├─► 拼接 ──► Transformer blocks ──► 预测结果
学习到的 mask tokens ──► 加入位置编码 ─┘
```

**关键设计：**

- **Mask Tokens**：可学习的参数（`nn.ParameterList`），每个数据集/遮罩模式可拥有独立的 mask token
- **多数据集支持**：通过 `PredictorMultiSeqWrapper` 处理不同帧率的数据
- **V2.1 变化**：predictor 也变得更深（V2 默认 depth=6，V2.1 默认 depth=24），同样有 hierarchical layers 输出
- **V2.1 `predict_all` 模式**：返回两组输出——pred（被遮罩 tokens）和 context（可见 tokens），分别用不同的投影头

#### (c) Target Encoder（目标编码器）—— 稳定训练的锚

- 是 Encoder 的**深拷贝**
- 通过 **EMA（指数移动平均）** 从 online encoder 缓慢更新
- 不参与梯度计算（`requires_grad = False`）
- EMA 更新公式：`param_k = m * param_k + (1 - m) * param_q`

```python
# app/vjepa/train.py 中实际代码
torch._foreach_mul_(params_k, m)          # param_k *= m
torch._foreach_add_(params_k, params_q, alpha=1 - m)  # param_k += (1-m) * param_q
```

EMA 调度从 0.99925 到 0.99925（即恒定），非常缓慢地跟踪 online encoder。

---

### 3.2 遮罩策略 —— 核心创新之一

**文件：** `src/masks/multiseq_multiblock3d.py`

V-JEPA 使用 **多块时空遮罩（Multi-block 3D Masking）**：

1. 在 3D 网格（时间×高×宽）上随机采样**多个矩形块**
2. 这些块内的 patch 被标记为「需要预测的」（pred mask）
3. 块外的 patch 是「可见的/上下文」（enc mask）

**生成过程（`_MaskGenerator.__call__`）：**

```
1. 随机采样块大小（时间尺度 × 空间尺度 × 宽高比）
2. 在 3D 空间随机放置 npred 个遮罩块
3. 合并所有遮罩块 → 得到 pred mask
4. 补集 → 得到 enc mask
5. 截断到统一的 min_keep 长度以形成 batch
```

以 ViT-g/16 配置为例，使用**两层遮罩**：

| 层级 | 块数 | spatial_scale | temporal_scale | 作用 |
|------|------|---------------|----------------|------|
| 第一层 | 8 个小块 | 0.15（各覆盖15%面积） | [1.0, 1.0]（全时域） | 学习局部时空模式 |
| 第二层 | 2 个大块 | 0.7（覆盖70%面积） | [1.0, 1.0]（全时域） | 学习全局依赖 |

这种多尺度、多块策略迫使模型：
- 从局部可见区域推断被遮挡的时空信息
- 学习远距离的时空依赖关系
- 理解物体的运动轨迹和变化

**可选模式：**
- `full_complement=True`：pred mask = enc mask 的补集（互斥）
- `inv_block=True`：反转角色——预测可见块，保留遮罩块作为上下文

---

### 3.3 训练目标（Loss）

**V-JEPA 2 的 Loss**（`app/vjepa/train.py`）：

```python
def loss_fn(z, h):
    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
    loss, n = 0, 0
    for zi, hi in zip(z, h):
        for zij, hij in zip(zi, hi):
            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    loss /= n
    return loss
```

就是**L1 损失**在特征空间（`loss_exp=1.0`），predictor 只对「被遮罩 token」做预测。

**V-JEPA 2.1 的改进——Dense Predictive Loss**（`app/vjepa_2_1/train.py`）：

V2.1 引入了**上下文损失（context loss）**，对所有 token 施加预测损失：

```python
loss = loss_pred + lambda * loss_context
```

其中 `loss_context` 的特征是对**可见 token** 也施加预测损失，但根据 token 到最近被遮罩块的**距离加权**（`compute_mask_distance`）：

```python
# 对每个可见 token，计算它到最近遮罩 token 的时空距离
# 距离越近权重越高（d^-0.5）
# 这让模型更关注遮罩边界附近的区域，产生更平滑、稠密的特征
```

另外 V2.1 支持**渐进式 lambda 调度**：训练初期 lambda 从小开始逐渐增大（`Lambda_LinearWarmupHold`）。

V2.1 还增加了**异常 loss 跳过机制**：如果 loss 超过均值+`loss_reg_std_mult`×标准差，则跳过该 step（不更新参数），防止训练不稳定。

---

### 3.4 V-JEPA 2-AC：动作条件世界模型

**文件：** `src/models/ac_predictor.py`

这是将 V-JEPA 2 用作**机器人世界模型**的扩展。在训练好的 V-JEPA 2 encoder 基础上，替换 predictor 为**动作条件 predictor**，用少量机器人交互数据微调。

**模型架构：**

```
每帧 tokens = [action_token, state_token, (extrinsics_token), frame_tokens...]
                                             |
                                             帧因果注意力掩码
                                             |
                                      predictor Transformer
```

**关键差异：**

| 特性 | V-JEPA 2 Predictor | V-JEPA 2-AC Predictor |
|------|---------------------|-----------------------|
| 输入 | mask tokens + frame tokens | action tokens + state tokens + frame tokens |
| 注意力 | 双向全连接 | **帧因果**（只能看过去） |
| 位置编码 | RoPE 或 sincos | RoPE（3D 分离旋转） |
| 输出 | 预测被遮罩特征 | 预测下一帧特征 |
| 额外编码器 | 无 | action_encoder, state_encoder, extrinsics_encoder |

**帧因果注意力掩码：**

```python
# 只允许当前帧关注历史帧
# build_action_block_causal_attention_mask(T, H, W, add_tokens)
# 生成一个上三角为 0 的稀疏掩码矩阵
```

**世界模型用法**（见 `notebooks/energy_landscape_example.ipynb` + `notebooks/utils/mpc_utils.py`）：

1. 给定当前帧观察和目标帧图像
2. 编码到特征空间
3. 在**动作空间**中进行采样/优化
4. 用 predictor 计算「预测特征 vs 目标特征」的误差作为**能量**
5. 选择能量最低的动作序列
6. 执行第一个动作，重复

这本质上是 **MPC（Model Predictive Control）** 在隐空间中的实现。

---

### 3.5 Attentive Probe：评估方案

**文件：** `src/models/attentive_pooler.py`

下游评估时，用 **AttentivePooler** 在冻结的 backbone 特征上训练一个轻量级分类头：

```python
# 1. 可学习的 query tokens（如 num_queries=1）
# 2. 与视频特征做 cross-attention（AttentivePooler）
# 3. MLP 可选（depth > 1）
# 4. 输出通过 Linear 分类（AttentiveClassifier）
```

比简单的全局平均池化更灵活，能关注到视频中最具判别力的时空区域。

**评估结果：**

| Benchmark | V-JEPA 2 (ViT-g/384) | 之前的 SOTA |
|-----------|----------------------|-------------|
| EK100 (动作预测) | 39.7% | 27.6% (PlausiVL) |
| SSv2 (视频分类) | 77.3% | 69.7% (InternVideo2-1B) |
| Diving48 (视频分类) | 90.2% | 86.4% (InternVideo2-1B) |
| MVP (Video QA) | 44.5% | 39.9% (InternVL-2.5) |
| TempCompass (Video QA) | 76.9% | 75.3% (Tarsier 2) |

---

## 四、训练细节

### 4.1 数据配置

**配置来源：** `configs/train/vitg16/pretrain-256px-16f.yaml`

- 多数据集联合训练，带权重采样
- 所有数据统一处理为 4fps，16帧（即 4 秒的 clip）
- 数据增强：随机裁剪（0.3~1.0 缩放，3/4~4/3 宽高比）、随机水平翻转
- 不使用 AutoAugment
- 分辨率：V2 用 256px，V2.1 用 384px

**V2.1 新增**：同时用 ImageNet-1K 图片联合训练。50% 的 GPU 处理图片、50% 处理视频。图片被当作 1 帧的视频处理（`img_temporal_dim_size=1`），使用独立的遮罩策略。

```yaml
# V2.1 配置中的图片数据部分
img_data:
  batch_size: 72
  crop_size: 256
  dataset_type: VideoDataset  # 复用视频加载器
  rank_ratio: 0.5            # 50% GPU 处理图片
  datasets:
    - /your_data/imagenet1k.csv
```

### 4.2 优化配置

| 参数 | V-JEPA 2 (ViT-g) | V-JEPA 2.1 (ViT-G) |
|------|------------------|-------------------|
| 优化器 | AdamW | AdamW |
| 学习率 | 5.25e-4 | 6e-4 |
| 权重衰减 | 0.04 → 0.04 (cosine) | 同左 |
| Warmup | 40 epochs | 40 epochs |
| 总轮数 | 800 | 1000 |
| EMA | 0.99925（恒定） | 同左 |
| Batch size/GPU | 24 | 24 |
| GPU 总数 | 128 (16 nodes × 8) | 128 |
| Precision | bfloat16 | bfloat16 |

**学习率调度（`src/utils/schedulers.py`）：**
```python
# WarmupCosineSchedule: 前40个epoch从start_lr线性升到ref_lr
# 之后cosine衰减到final_lr
# ipe_scale=1.25 延长总步数
```

**权重衰减调度（`CosineWDSchedule`）：**
```python
# 余弦调度从 weight_decay 衰减到 final_weight_decay
```

**参数分组：**
```python
# 权重参数（weight）→ 应用 weight_decay
# 偏置参数（bias）和 norm 参数（len(shape)==1）→ weight_decay=0
```

### 4.3 分布式训练

- `main_distributed.py` = SLURM 启动 + `srun python -m app.main`
- `DistributedDataParallel` 包装所有模型
- Encoder 和 Target Encoder 使用 `static_graph=True`（加速）
- Predictor 使用 `static_graph=False, find_unused_parameters=True`
- 使用 activation checkpointing 管理显存

**完整训练命令：**
```bash
# 本地
python -m app.main --fname configs/train/vitg16/pretrain-256px-16f.yaml --devices cuda:0

# 分布式（SLURM）
python -m app.main_distributed \
  --fname configs/train/vitg16/pretrain-256px-16f.yaml \
  --time 6000 --account my_account
```

---

## 五、代码组织架构

```
vjepa2/
│
├── app/                              # 训练入口和循环
│   ├── main.py                       #   本地训练入口
│   ├── main_distributed.py           #   SLURM分布式训练入口
│   ├── scaffold.py                   #   训练启动脚手架
│   │
│   ├── vjepa/                        #   V-JEPA 2 预训练
│   │   ├── train.py                  #     训练主循环
│   │   ├── utils.py                  #     模型初始化、优化器配置、checkpoint加载
│   │   └── transforms.py             #     视频数据增强
│   │
│   ├── vjepa_2_1/                    #   V-JEPA 2.1 预训练
│   │   ├── train.py                  #     增强版训练循环
│   │   ├── utils.py                  #     工具函数
│   │   ├── transforms.py             #     数据增强
│   │   ├── wrappers.py               #     多数据集包装器
│   │   └── models/
│   │       ├── vision_transformer.py #     带hierarchical输出的ViT
│   │       ├── predictor.py          #     带多级输出的Predictor
│   │       └── utils/
│   │           ├── masks_dist.py     #     距离加权掩码计算
│   │           ├── modules.py        #     Block、Attention等组件
│   │           ├── patch_embed.py    #     2D/3D patch embed
│   │           └── pos_embs.py       #     位置编码
│   │
│   └── vjepa_droid/                  #   动作条件训练（V-JEPA 2-AC）
│       ├── train.py                  #     训练循环
│       ├── transforms.py             #     数据增强
│       ├── utils.py                  #     工具函数
│       └── droid.py                  #     Droid数据集处理
│
├── src/                              # 核心库
│   ├── models/                       #   模型定义
│   │   ├── vision_transformer.py     #     ViT编码器（所有变体）
│   │   ├── predictor.py              #     特征预测器
│   │   ├── ac_predictor.py           #     动作条件预测器
│   │   ├── attentive_pooler.py       #     评估用注意力池化
│   │   └── utils/
│   │       ├── modules.py            #     Attention/MLP/Block/RoPE等组件
│   │       ├── patch_embed.py        #     Conv2D/Conv3D patch嵌入
│   │       └── pos_embs.py           #     2D/3D sincos位置编码
│   │
│   ├── masks/                        #   遮罩策略
│   │   ├── default.py                #     默认collator（无遮罩）
│   │   ├── multiseq_multiblock3d.py  #     3D多块遮罩（核心）
│   │   └── utils.py                  #     apply_masks工具
│   │
│   ├── datasets/                     #   数据处理
│   │   ├── data_manager.py           #     数据加载器工厂
│   │   ├── video_dataset.py          #     视频数据集
│   │   ├── imagenet1k.py             #     ImageNet数据集
│   │   └── utils/
│   │       ├── video/                #     视频预处理
│   │       │   ├── transforms.py     #     自定义transform
│   │       │   ├── transforms_builder.py #   transform构建
│   │       │   ├── functional.py     #     函数式视频处理
│   │       │   ├── volume_transforms.py  # 3D transform
│   │       │   ├── randaugment.py    #     RandAugment
│   │       │   └── randaerase.py     #     Random Erasing
│   │       ├── dataloader.py         #     DataLoader包装
│   │       ├── weighted_sampler.py   #     加权采样
│   │       └── worker_init_fn.py     #     worker初始化
│   │
│   └── utils/                        #   工具函数
│       ├── distributed.py            #     分布式训练工具
│       ├── logging.py                #     日志记录
│       ├── schedulers.py             #     学习率调度器
│       ├── tensors.py                #     张量操作工具
│       ├── wrappers.py               #     MultiSeqWrapper
│       ├── checkpoint_loader.py      #     checkpoint加载
│       └── monitoring.py             #     训练监控
│
├── evals/                            # 评估代码
│   ├── main.py                       #   本地评估入口
│   ├── main_distributed.py           #   分布式评估入口
│   ├── scaffold.py                   #   评估脚手架
│   │
│   ├── video_classification_frozen/  #   视频分类评估
│   │   ├── eval.py                   #     训练/评估循环
│   │   ├── models.py                 #     模型配置
│   │   ├── utils.py                  #     工具函数
│   │   └── modelcustom/              #     自定义模型适配
│   │
│   ├── action_anticipation_frozen/   #   动作预测评估
│   │   └── ...                       #     (同上结构)
│   │
│   └── image_classification_frozen/  #   图片分类评估
│   │   └── ...                       #     (同上结构)
│   │
│   └── hub/                          #   PyTorch Hub入口
│       ├── __init__.py
│       └── preprocessor.py           #     视频预处理器
│
├── configs/                          # YAML 配置文件
│   ├── train/                        #   V2 预训练/cooldown/droid
│   │   ├── vitl16/
│   │   ├── vith16/
│   │   ├── vitg16/                   #     1B模型配置
│   │   └── ...
│   ├── train_2_1/                    #   V2.1预训练
│   │   ├── vitb16/
│   │   ├── vitl16/
│   │   ├── vitg16/                   #     1B模型
│   │   └── vitG16/                   #     2B模型
│   ├── eval/                         #   评估配置
│   ├── inference/                    #   推理配置
│   └── ...
│
├── notebooks/                        # Jupyter notebook & 脚本
│   ├── vjepa2_demo.ipynb             #   使用demo
│   ├── vjepa2_demo.py                #   Python脚本版本
│   ├── energy_landscape_example.ipynb #   AC模型可视化
│   └── utils/
│       ├── world_model_wrapper.py    #   世界模型包装器
│       └── mpc_utils.py             #   MPC规划工具
│
├── tests/                            # 单元测试
├── hubconf.py                        # PyTorch Hub配置
├── setup.py                          # Python包配置
└── README.md
```

### 关键设计模式：`MultiSeqWrapper`

**文件：** `src/utils/wrappers.py`

这是连接训练框架与模型的核心设计。它的作用是支持**不同帧率/不同遮罩策略的多个数据集同时训练**：

```python
class MultiSeqWrapper(nn.Module):
    def forward(self, x, masks=None):
        # x = [batch_fpc16, batch_fpc32]  # 多数据集列表
        # masks = [[mask_set_1], [mask_set_2]]  # 每组的遮罩
        
        outs = [[] for _ in x]
        for i, (xi, mi) in enumerate(zip(x, masks)):
            for mij in mi:
                outs[i] += [self.backbone(xi, masks=mij)]
        return outs
        # 返回 = [[fpc16_mask1_out, fpc16_mask2_out], [fpc32_mask1_out]]
```

每个数据集可以有完全不同的帧率、分辨率、遮罩策略，但共享同一个 backbone。

---

## 六、V2 vs V2.1 核心差异

| 方面 | V-JEPA 2 | V-JEPA 2.1 |
|------|----------|------------|
| **损失函数** | 只预测被遮罩 token | 预测所有 token（dense predictive loss） |
| **特征层级** | 只输出最后一层 | 输出 4 个 hierarchical 层，拼接 |
| **Predictor 深度** | 6 层 | 24 层 |
| **Predictor 输入** | 单级特征 | 多级级联特征（embed_dim×4 → embed_dim → pred_dim） |
| **多模态** | 仅视频 | 视频 + 图片联合训练（各占 50% GPU） |
| **上下文损失** | 无 | 有，距离加权 |
| **Loss 调节** | 无 | 渐进式 lambda + 异常 loss 跳过 |
| **最大模型** | ViT-g (1B, 40层) | ViT-G (2B, 48层) |
| **训练分辨率** | 256px | 384px |

---

## 七、上手使用

### 加载预训练模型

**PyTorch Hub：**
```python
import torch

# 预处理器
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')

# V-JEPA 2 模型
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant_384')

# V-JEPA 2.1 模型
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_gigantic_384')

# V-JEPA 2-AC（两个组件）
encoder, ac_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
```

**HuggingFace：**
```python
from transformers import AutoVideoProcessor, AutoModel

model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
```

### 运行评估
```bash
# 训练 attentive probe
python -m evals.main --fname configs/eval/vitg-384/ssv2.yaml --devices cuda:0

# 分布式
python -m evals.main_distributed --fname configs/eval/vitg-384/ssv2.yaml
```

### 预训练
```bash
# 本地
python -m app.main --fname configs/train/vitg16/pretrain-256px-16f.yaml --devices cuda:0

# V2.1
python -m app.main --fname configs/train_2_1/vitG16/pretrain-256px-16f.yaml --devices cuda:0
```

### 机器人后训练
```bash
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml --devices cuda:0
```

### Demo
```bash
# 下载 checkpoint
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt

# 修改 vjepa2_demo.py 中的路径后运行
python -m notebooks.vjepa2_demo
```

### Notebooks
- `notebooks/vjepa2_demo.ipynb`：完整的推理 demo
- `notebooks/energy_landscape_example.ipynb`：V-JEPA 2-AC 能量景观可视化

---

## 八、理解与评价

### 设计哲学

V-JEPA 2 是 **Yann LeCun「世界模型」愿景**的一个重要实践。其核心设计理念：

1. **预测即理解（Prediction as Understanding）**：通过预测被遮挡的时空区域，模型被迫学习物理世界的 motion、interaction 和 temporal dynamics
2. **隐空间预测 > 像素预测**：在特征空间做预测迫使模型抽象出高层次语义，而不是记忆像素细节（与 MAE 的本质区别）
3. **自监督 > 监督**：不需要任何人工标注，直接从海量视频中学习，可无限扩展
4. **世界模型 > 识别模型**：V-JEPA 2-AC 展示了学到的表征可以做**规划**（planning），而不仅仅是识别

### V2.1 改进的意义

V2.1 的 Dense Predictive Loss + Deep Supervision + 多模态训练，解决了 V2 在**稠密预测任务**（如分割、深度估计）上不够强的问题。核心洞察是：只预测被遮罩 token 会忽略「已知区域」（context tokens）的特征质量；通过对所有位置施加预测损失（即使可见的也需自我重建），迫使模型在每处都产生有意义的表示。

### 代码质量评价

- **组织清晰**：app（训练）/ src（模型）/ evals（评估）三层分离
- **配置驱动**：所有超参数通过 YAML 配置，无需硬编码
- **分布式友好**：从本地到 SLURM 集群无缝切换
- **多数据集支持**：`MultiSeqWrapper` + `MaskCollator` 的设计优雅
- **注意**：部分组件（如 V2.1 的 `VisionTransformer`）已经重写，与 V2 不完全兼容

### 可能的改进方向

- V2.1 的 hierarchical 输出拼接策略（`cat([feat_4, feat_8, feat_16, feat_32])`）会增大维度，是否有更高效的融合方式？
- 当前 mask 策略全数据统一，是否可以对不同数据集自适应调整 mask 分布？
- V-JEPA 2-AC 目前只验证了 Franka 机械臂，是否可以扩展到更通用的机器人平台？
