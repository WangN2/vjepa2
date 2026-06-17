# V-JEPA 2-AC 讨论总结

> 围绕 V-JEPA 2-AC 方案展开的技术讨论：架构原理、推理耗时、CEM 机制、自动驾驶适配方案。

---

## 1. 方案定位

V-JEPA 2-AC 是 V-JEPA 2 论文中的 Action-Conditioned 变体。在 V-JEPA 2 预训练 ViT 编码器基础上，用 DROID 机器人轨迹数据后训练一个动作条件化的 latent world model。

### 论文与资源

- **论文**: [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)，arXiv `2506.09985`，Meta FAIR
- **代码 & 权重**: [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- **数据效率**: 预训练 100 万+ 小时互联网视频（自监督），AC 后训练仅需 ~62 小时 DROID 机器人视频
- **零样本规划**: Franka Panda 机械臂上 Pick-and-Place，Cup 80%、Box 65%（Cosmos 和 Octo 均为 0-15%）

---

## 2. 模型结构

两个独立网络组成：Encoder（冻结）+ AC Predictor（可训练）。

### 2.1 Encoder — ViT-g/16（冻结）

```
输入: video clip [B, 3, T, 256, 256]
  → PatchEmbed3D (conv3d, tubelet=2)
  → [B, T/2*256, 1408] tokens
  → Block × 40 (RoPEAttention + SwiGLU FFN + DropPath)
  → [B, T/2*256, 1408]

参数: ~1.1B, embed_dim=1408, 22 heads, 3D RoPE
训练状态: 冻结 (EMA 更新 target_encoder)
```

### 2.2 AC Predictor（可训练）

```
输入: patch tokens [B, T*256, 1408] + actions [B, T-1, 7] + states [B, T, 7]

 ┌─ Embedding Projections ───────────────────────┐
 │  predictor_embed:    Linear(1408 → 1024)       │
 │  action_encoder:     Linear(7    → 1024)       │
 │  state_encoder:      Linear(7    → 1024)       │
 │  extrinsics_encoder: Linear(6    → 1024)       │
 └────────────────────────────────────────────────┘
                        │
 ┌─ Token Interleaving (每帧) ────────────────────┐
 │  Frame_t = [action_t | state_t | patches_t×256]│
 │  concat → [B, T*(2+256), 1024]                 │
 └────────────────────────────────────────────────┘
                        │
 ACBlock × 24 (ACRoPEAttention + SwiGLU FFN)
   - Action tokens: 仅时间 RoPE
   - Patch tokens:  3D RoPE (T+H+W)
   - Block-Causal Mask: 帧 t 只能看到帧 0..t
                        │
 ┌─ Output ───────────────────────────────────────┐
 │  丢弃 action/state tokens                       │
 │  保留 patch tokens → predictor_norm → proj      │
 │  → [B, T*256, 1408]                            │
 └────────────────────────────────────────────────┘

参数: ~300M, embed_dim=1024, depth=24, 16 heads
训练状态: 可训练
```

### 2.3 Encoder vs Predictor 对比

| | Encoder | AC Predictor |
|---|---|---|
| embed_dim | 1408 | 1024 |
| depth | 40 | 24 |
| heads | 22 | 16 |
| Attention | 双向 | Block-Causal (帧级因果) |
| RoPE | 3D (T,H,W) | 3D for patches, 1D(T) for actions |
| 训练状态 | 冻结 | 可训练 |

### 2.4 ACRoPEAttention — 核心区别

与普通 RoPEAttention 的关键不同：

1. 输入序列中 action token 和 patch token **交错排列**
2. **分开处理**: action token 展平后单独算 QKV，仅做时间维 RoPE；patch token 正常做 3D RoPE
3. **合并**: action QKV 和 patch QKV 拼回同一序列，用 block-causal mask 做 SDPA

---

## 3. 训练阶段

训练 **不使用 CEM**，是简单的监督学习。

```python
# 训练循环核心逻辑 (app/vjepa_droid/train.py)

# 1. Target Encoder: 编码全帧得到 GT latent
h = target_encoder(clips)                    # [B, T*256, 1408]

# 2. AC Predictor: 给定历史帧 + 真实动作 → 预测下一帧
z_tf = predictor(h[:, :-256], actions, states[:, :-1])

# 3. Teacher-forced loss
jloss = L1(z_tf, h[:, 256:])

# 4. Auto-regressive rollout (auto_steps=2)
# 用预测的 latent 继续展开一步
z_ar = autoregressive_rollout(predictor, h[:, :256], z_tf[:, :256], actions)
sloss = L1(z_ar, h[:, 256:])

# 5. 总 loss
loss = jloss + sloss
loss.backward()
```

**训练只做一件事**: 让 Predictor 学会映射 `(latent, action) → next_latent`。动作来自数据集提供的真实动作，不涉及任何搜索。

### DROID 数据格式

```
CSV → 每行一个轨迹目录
  轨迹目录/
    ├── trajectory.h5    (robot_state, camera_extrinsics, gripper_position)
    └── recordings/MP4/  (left/right/wrist 相机视角的 mp4)

- 动作 (7D): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]
- 状态 (7D): [x, y, z, roll, pitch, yaw, gripper]
- 外参 (6D): 相机在机器人坐标系下的位姿
```

---

## 4. CEM (Cross-Entropy Method)

### 4.1 是什么

一种**基于采样的无梯度优化算法**。在无法求梯度的情况下，通过反复采样-评估-精英筛选来逼近最优解。

**直观类比 — 投篮**:

```
第 1 轮: 随机尝试 400 种角度/力度 → 保留命中最准的 40 个
第 2 轮: 围绕这 40 个精英的均值/方差再采样 400 个
...
第 15 轮: 取最终分布的均值作为最佳动作
```

### 4.2 在 V-JEPA 2-AC 中的流程

```python
mean = 0, std = maxnorm

for i in range(15):                        # CEM 迭代
    # 1. 从当前高斯分布采样 400 条动作序列
    actions = randn(400, 16, 3) * std + mean

    # 2. 每条序列用 AC Predictor 展开 16 步
    for each sequence:
        latent_1 = predictor(latent_0, a_0)
        latent_2 = predictor(latent_1, a_1)
        ...
        energy = L1(latent_16, goal_latent)   # 与目标 latent 的距离

    # 3. 取能量最低的 40 条
    elites = topk(energy, k=40)

    # 4. 用精英更新采样分布 (EMA)
    mean = 0.75 * old_mean + 0.25 * mean(elites)
    std  = 0.95 * old_std  + 0.05 * std(elites)

return mean  # 最终均值序列的第一个动作
```

### 4.3 CEM 只在推理阶段使用

| | 训练 | 推理 |
|---|---|---|
| 有动作吗 | 有，数据集提供 | 没有，CEM 搜索出来 |
| Predictor 做什么 | 学习 `(latent,action)→next_latent` | 被 CEM 反复调用做 rollout |
| 优化方式 | 梯度下降 | CEM 无梯度采样搜索 |
| 耗时 | 不要求实时 | ~16s |

---

## 5. 推理耗时分析

### 5.1 端到端耗时（图像 → 动作）

```
Step 1: 编码当前帧 (ViT-g, 40层)        ~15ms
Step 2: 编码目标帧 (ViT-g, 40层)        ~15ms
Step 3: CEM 规划 (15 iter × 16 rollout)  ~16,000ms
────────────────────────────────────────────────
总计                                      ~16s / 动作
```

**瓶颈**: CEM 需要串行调用 Predictor 240 次 (batch=400, sequence 1→16 帧递增)，占总耗时 99.8%。

### 5.2 单次 Predictor Forward 耗时估算 (A100, bfloat16)

| 序列长度 | Tokens (batch=400) | 耗时 |
|---------|---------------------|------|
| T=1 (258 tok) | 103K | ~5ms |
| T=8 (2064 tok) | 826K | ~30ms |
| T=16 (4128 tok) | 1.65M | ~60ms |

### 5.3 为什么比 Cosmos 快 16×

| | V-JEPA 2-AC | Cosmos |
|---|---|---|
| 操作空间 | Latent (1408-d) | Pixel (视频帧生成) |
| 预测器 | 24层 Transformer, ~300M | Diffusion 模型, 多步去噪 |
| 单动作规划 | ~16s | ~4min |

本质原因：在紧凑的 latent 空间规划，不需要生成像素。

### 5.4 优化方向

| 优化 | 效果 | 备注 |
|------|------|------|
| KV-cache 增量推理 | 16s → 1-2s | 避免每步重算完整序列 |
| 批量并行 rollout | 大幅减少 kernel launch | 利用 block-causal mask |
| 减小 CEM 参数 (50样×5轮) | 16s → 0.5-1s | 牺牲规划质量 |
| 换小模型 (ViT-B, pred depth=4) | 单次 forward ~0.3ms | 可能损失精度 |
| torch.compile / CUDA graph | 20-30% 提升 | 消除 kernel launch overhead |

---

## 6. 适配自动驾驶

### 6.1 输入输出对比

| | 当前 V-JEPA 2-AC | 自动驾驶需求 |
|---|---|---|
| 视觉输入 | 单目相机 | 前视相机（可能多视角） |
| 动作空间 | 7-DoF 机械臂 | 2-DoF (steer, throttle) |
| 状态空间 | 7-DoF 关节状态 | 自车状态 (~20维) |
| 指令 | 无（goal image） | 语言指令 "去沙发" |
| 输出 | 机械臂动作序列 | 轨迹 waypoints / 控制量 |
| 实时要求 | 不强制 | <100ms |

### 6.2 两条技术路线

#### 路线 A: Policy Head（推荐）

```
camera image → Encoder(冻结) → visual tokens ─┐
instruction  → LangEncoder    → lang_token   ─┼→ Fusion Transformer → 轨迹 head → waypoints
ego_state    → StateEncoder   → state_token  ─┘

耗时: ~20ms ✓
```

把 AC Predictor 的动作条件化机制拆掉，改成 policy head 直接输出轨迹。

#### 路线 B: World Model + 极简规划

```
camera image → Encoder → latent_now ─┐
instruction  → Lang2Latent → goal    ┤→ 极简 CEM/MPC → 动作
                                      │  (50 samples, 5 iter,
                                      │   batch rollout)
耗时: ~30-50ms (优化后)
```

保留 AC Predictor 作为 world model，用极简搜索输出动作。

### 6.3 改造要点

| 模块 | 当前 | 改造后 |
|------|------|--------|
| Encoder | ViT-g (1.1B) | ViT-B/L (100-300M) |
| Action encoder | Linear(7, 1024) | Linear(2, 1024) |
| State encoder | Linear(7, 1024) | Linear(ego_dim, 1024) |
| Predictor depth | 24 | 4-6 |
| 新增 | — | Language Encoder |
| 新增 | — | Trajectory Head (路线 A) |
| CEM | samples=400, iter=15 | 不需要 (路线 A) / 极简 (路线 B) |

---

## 7. Goal Latent 问题

### 7.1 问题描述

CEM 需要一个 goal latent 来评估 "哪条轨迹最好"：

```
energy = ‖ predictor_rollout(final_state) — goal_latent ‖
```

论文场景中 goal 来自**拍一张目标位置照片**（机械臂推到目标位姿拍照）。但在 "去沙发那里" 的语言指令场景，**没有沙发视角的照片**。

### 7.2 三种解决方案

#### 方案 1: Language → Latent 直接映射

```
"去沙发那里" → Language2Latent 模型 → goal_latent

训练: 收集 (goal_image, instruction) 对
      goal_image → Encoder → GT latent
      训练 Language2Latent 回归 GT latent
优势: 快 (~2ms)，直接在 latent 空间
劣势: 需要配对数据
```

#### 方案 2: Language-Conditioned Predictor + 评分

```
当前: predictor(latent, action) → next_latent
改造: predictor(latent, action, lang_token) → next_latent

CEM 目标函数改为:
  score = alignment(latent, lang_token)  # CLIP 式评分
```

#### 方案 3: MPC + 轨迹层面评分（自动驾驶推荐）

```
采样 N 条候选轨迹 → AC Predictor 展开 → 评分:
  ✓ 安全性 (不碰撞)
  ✓ 舒适性 (加速度/jerk)
  ✓ 遵守交通规则
  ✓ 对齐导航指令
→ 选最优轨迹 → 执行第一步

# goal 不是 latent 点，而是一组约束条件
# World Model 不变，只换上层规划器的目标函数
```

### 7.3 方案对比

| 方案 | Goal 来源 | 适合场景 | 改造量 |
|------|----------|---------|--------|
| Language → Latent | 训练 text2latent | 室内导航、目标明确 | 中等 |
| Lang-Conditioned Pred + 评分 | CLIP 式对齐 | 通用指令跟随 | 较大 |
| MPC + 轨迹评分 | 约束条件集合 | **自动驾驶** | 较小 |

---

## 8. 核心文件索引

| 文件 | 作用 |
|------|------|
| `src/models/ac_predictor.py` | AC Predictor 模型定义 |
| `src/models/utils/modules.py` | ACBlock, ACRoPEAttention, block-causal mask |
| `src/models/vision_transformer.py` | Encoder (复用 V-JEPA 2) |
| `app/vjepa_droid/train.py` | 训练主循环 |
| `app/vjepa_droid/droid.py` | DROID 数据集加载 |
| `app/vjepa_droid/utils.py` | 模型初始化、optimizer |
| `configs/train/vitg16/droid-256px-8f.yaml` | 训练配置 |
| `notebooks/nav_planning/nav_mpc.py` | CEM 规划器实现 |
| `notebooks/nav_planning/nav_world_model.py` | World Model wrapper |
| `src/hub/backbones.py` | PyTorch Hub 入口 |
