# V-JEPA 2-AC 方案速览

> Action-Conditioned：在 V-JEPA 2 预训练编码器基础上，用机器人数据后训练一个动作条件化的世界模型。

## 0. 论文与资源

- **论文**: [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
  - arXiv: `2506.09985`, 2025-06-11, 48 pages
  - 作者: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Yann LeCun 等 (Meta FAIR)
  - 同一篇论文涵盖 V-JEPA 2（预训练）、V-JEPA 2.1（dense 特征）和 V-JEPA 2-AC（动作条件化世界模型）
- **官方博客**: [ai.meta.com/blog/v-jepa-2-world-model-benchmarks](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)
- **代码 & 权重**: [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- **Checkpoint**: `vjepa2-ac-vitg.pt`, 通过 PyTorch Hub 加载: `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')`

### 核心亮点

- **数据效率极高**: V-JEPA 2 在 100 万+ 小时互联网视频上自监督预训练，V-JEPA 2-AC 仅需 **~62 小时无标注 DROID 机器人视频** 后训练
- **零样本规划**: 无需环境特定数据采集、任务特定训练或 reward 设计，直接在 Franka Panda 机械臂上完成 reaching、grasping、pick-and-place 任务
- **规划速度**: 比 diffusion-based 方案 (Cosmos) 快 **16 倍**（~16s/action vs ~4min/action），使用 Cross-Entropy Method (CEM) 在 latent 空间优化动作序列
- **Pick-and-Place 成功率**:

  | Method | Cup | Box |
  |--------|-----|-----|
  | Octo | 15% | 10% |
  | Cosmos | 0% | 0% |
  | **V-JEPA 2-AC** | **80%** | **65%** |

### 规划范式

```
给定目标图像 → Target Encoder 提取 goal latent
当前观测     → AC Predictor 展开候选动作序列的 latent rollout
              → CEM 优化: 最小化 rollout latent 与 goal latent 的距离
              → 执行最优动作序列的第一步 (MPC)
```

## 1. 一句话总结

V-JEPA 2 学会"看懂视频"，V-JEPA 2-AC 在此基础上学会"给定动作，预测未来"。它冻结 V-JEPA 2 的 ViT 编码器，训练一个接收动作/状态/外参的 AC Predictor，以 teacher-forcing + 自回归 rollout 方式预测下一帧的 latent 特征。

## 2. 与 V-JEPA 2 / 2.1 的核心区别

| | V-JEPA 2 | V-JEPA 2.1 | V-JEPA 2-AC |
|---|---|---|---|
| **编码器** | ViT (src/) | ViT + register + 多层级输出 (app/vjepa_2_1/) | 同 V-JEPA 2 编码器 |
| **预测器** | `VisionTransformerPredictor` | 多层蒸馏预测器 | `VisionTransformerPredictorAC` (全新) |
| **输入** | 视频 + [MASK] token | 视频 + [MASK] token | 视频 + 动作 + 状态 + 外参 |
| **预测目标** | 被 mask 位置的 latent 特征 | 多层 dense 特征 | 下一帧的 latent 特征 |
| **注意力** | 双向/因果可选 | 因果可选 | 帧级 block-causal（必须） |
| **训练数据** | 互联网视频 | 互联网视频 + 图像 | DROID 机器人轨迹 |
| **训练范式** | 自监督 | 自监督 | 后训练（监督学习） |

## 3. 架构数据流

```
输入: video clip [B, C, T, H, W]

  ┌─────────────────────────────────────────────────────┐
  │  Target Encoder (ViT-g, frozen EMA)                  │
  │  全帧编码 → target latents h [B, T, N_tokens, D]     │
  └─────────────────────────────────────────────────────┘
                           ↓ (h 作为 GT 监督)

  ┌─────────────────────────────────────────────────────┐
  │  AC Predictor (24-block ViT, 可训练)                 │
  │                                                       │
  │  每帧 token 序列: [action | state | extrin* | patches]│
  │  帧间: block-causal mask (只能看到当前及历史帧)      │
  │                                                       │
  │  Teacher-forced: 用真实 action 预测 frame 2           │
  │  Auto-regressive: 用预测 token 滚动预测 frame 3+     │
  └─────────────────────────────────────────────────────┘
                           ↓
  Loss = L1(z_tf, h) + L1(z_ar, h)  (均对后续帧计算)
```

## 4. 关键实现细节

### 4.1 AC Predictor (`src/models/ac_predictor.py`)

- 3 个线性编码器: `action_encoder(7→1024)`, `state_encoder(7→1024)`, `extrinsics_encoder(6→1024)`
- Token 交错: 每帧 = `[a_token, s_token, (e_token), H*W patch_tokens]`
- 24 层 `ACBlock`（带 RoPE 的因果注意力 + SwiGLU MLP）
- 输出: 仅保留 patch tokens，投影回 `embed_dim=1408`

### 4.2 Block-Causal Attention (`src/models/utils/modules.py`)

```python
def build_action_block_causal_attention_mask(T, H, W, add_tokens):
    # 每帧 N_T = add_tokens + H*W 个 token
    # mask[T1*N_T : (T1+1)*N_T, T2*N_T : (T2+1)*N_T] = 1  (仅当 T2 ≤ T1)
    # → 每帧可以看到自己及之前所有帧的全部 token
```

### 4.3 训练循环 (`app/vjepa_droid/train.py`)

`auto_steps=2` 意味着:
1. **Teacher-forced step**: 用 frame 1 的真实 token + frame 1→2 的真实 action 预测 frame 2
2. **Autoregressive step**: 用预测的 frame 2 token + frame 1→2→3 的真实 action 预测 frame 3

两个预测结果都参与 loss 计算。

### 4.4 数据格式 (DROID)

```
CSV 文件 → 每行一个轨迹目录
  轨迹目录/
    ├── trajectory.h5    (robot_state, camera_extrinsics, gripper_position)
    └── recordings/MP4/  (left/right/wrist 相机视角的 mp4)
```

- 动作 (7D): `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]`
- 状态 (7D): `[x, y, z, roll, pitch, yaw, gripper]`
- 外参 (6D): 相机在机器人坐标系下的位姿

## 5. 文件索引

| 文件 | 作用 |
|---|---|
| `src/models/ac_predictor.py` | AC 预测器模型定义 |
| `src/models/utils/modules.py` | `ACBlock`, `ACRoPEAttention`, causal mask |
| `src/models/vision_transformer.py` | 编码器（复用 V-JEPA 2） |
| `app/vjepa_droid/train.py` | 训练主循环 |
| `app/vjepa_droid/droid.py` | DROID 数据集加载 |
| `app/vjepa_droid/utils.py` | 模型初始化、optimizer、checkpoint |
| `app/vjepa_droid/transforms.py` | 数据增强 |
| `configs/train/vitg16/droid-256px-8f.yaml` | 训练配置（唯一） |
| `src/hub/backbones.py` | PyTorch Hub 入口 |

## 6. 训练命令

```bash
# 单机多卡
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml --devices cuda:0

# SLURM 分布式 (4 nodes × 8 GPUs)
python -m app.main_distributed --fname configs/train/vitg16/droid-256px-8f.yaml
```

## 7. 推理加载

```python
encoder, ac_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')

# encoder: 冻结的 ViT-g 编码器
# ac_predictor: 训练好的 AC 预测器
```

## 8. 关键超参数 (droid-256px-8f.yaml)

| 参数 | 值 |
|---|---|
| 编码器 | ViT-g/16 (embed_dim=1408, 40 blocks) |
| 预测器深度 | 24 blocks, embed_dim=1024, 16 heads |
| 分辨率 | 256×256, patch=16, tubelet=2 |
| 帧数 | 8 frames, 4 fps |
| Batch size | 8 per GPU |
| 优化器 | AdamW, lr=4.25e-4, wd=0.04 |
| 训练 | 315 epochs, 15 warmup + 15 anneal |
| Loss | L1 (loss_exp=1.0), teacher-forced + 1 autoregressive step |
| 精度 | bfloat16 mixed precision |
