# V-JEPA 2-AC 蒸馏部署方案：World Model → Onboard Policy

> 核心思路：云端用 World Model + CEM 做大规模规划生成专家轨迹，然后蒸馏 AC Predictor 为车载轨迹 Decoder，实现 <100ms 实时推理。

---

## 1. 问题背景

V-JEPA 2-AC 提供了两个组件：

- **Encoder**（ViT-g，冻结）：图像 → visual latent
- **AC Predictor**（24 层 Transformer）：`(latent, action, state) → next_latent`，即 World Model

World Model 本身不能直接输出动作，需要 CEM 在线搜索，耗时 ~16s/action，无法满足自动驾驶 <100ms 的实时要求。

---

## 2. 核心思路：Distillation from Planner（从规划器蒸馏）

分两个阶段：

```
阶段 1 — 云端（离线，不要求实时）:
  World Model + CEM 为海量场景生成高质量轨迹
  → 形成 "场景→轨迹" 数据集

阶段 2 — 车载（在线，<20ms）:
  AC Predictor + 轨迹 head → 直接输出轨迹
  → 一次 forward，不搜索
```

这是一种 **Expert Iteration** 范式：让 CEM 充当 "专家"，policy 通过模仿专家来学习。

---

## 3. 两阶段架构

### 3.1 阶段 1：云端 World Model 训练 + CEM 规划

```
                    ┌─────────────────────────┐
  海量场景数据      │                         │
  - 图像            │  ┌──────────────────┐   │
  - 导航目标        │  │  World Model     │   │
  - 语言指令        │  │  (AC Predictor)  │   │
  - 自车状态        │  │                  │   │
                    │  │  latent + action  │   │
                    │  │  → next_latent   │   │
                    │  └────────┬─────────┘   │
                    │           │              │
                    │           ▼              │
                    │  ┌──────────────────┐   │
                    │  │  CEM 规划器       │   │
                    │  │                  │   │
                    │  │  for i in 15:    │   │
                    │  │    采样 400 条   │   │
                    │  │    rollout 16 步 │   │
                    │  │    选 top-40     │   │
                    │  │  更新分布        │   │
                    │  │                  │   │
                    │  │  → 最优轨迹      │   │
                    │  └──────────────────┘   │
                    │                         │
                    └─────────────────────────┘
                              │
                              ▼
               数据集: (场景, 条件) → 专家轨迹
```

### 3.2 阶段 2：蒸馏 — AC Predictor → Trajectory Decoder

```
  ┌─────────────────────────────────────────────────┐
  │              Trajectory Decoder                  │
  │              （车载部署）                         │
  │                                                  │
  │  图像 ──→ Encoder (冻结) ──→ visual latent       │
  │                                  │               │
  │  导航目标 ──→ Goal Encoder      │               │
  │  语言指令 ──→ Lang Encoder      │               │
  │  自车状态 ──→ State Encoder     │               │
  │                                  │               │
  │        ┌─────────────────────────┘              │
  │        ▼                                         │
  │  ┌─────────────────────────────┐                │
  │  │  Decoder Backbone           │                │
  │  │  (= AC Predictor 结构复用)   │                │
  │  │  24 层, block-causal        │                │
  │  │  attention, 3D RoPE         │                │
  │  │                              │                │
  │  │  输入: [goal|lang|state|     │                │
  │  │         patches] per frame   │                │
  │  └──────────────┬──────────────┘                │
  │                 ▼                                │
  │  ┌─────────────────────────────┐                │
  │  │  Trajectory Head (新增)      │                │
  │  │  Linear → SiLU → Linear     │                │
  │  │  → waypoints [20, 2]        │                │
  │  └─────────────────────────────┘                │
  │                                                  │
  │  耗时: ~20ms（一次 forward）                     │
  └─────────────────────────────────────────────────┘
```

---

## 4. 为什么 AC Predictor 可以变成 Decoder

| | AC Predictor (原) | Trajectory Decoder (蒸馏后) |
|---|---|---|
| 训练任务 | `(latent, action) → next_latent` | `(scene, conditions) → trajectory` |
| 结构 | 24 层 Transformer, block-causal | **完全复用结构** |
| 输出头 | `proj: 1024→1408`（预测 latent） | `traj_head: 1024→40`（预测轨迹） |
| 预训练 | World Model 动力学 | **从 World Model backbone 初始化** |
| 能力来源 | 学 "做什么会发生什么" | 继承了对动力学的理解 |

AC Predictor 作为 Decoder 的核心优势：

1. **Block-causal attention 天然适合序列预测**：当前帧看到历史，不能看未来，正是解码轨迹所需的时序归纳偏置
2. **3D RoPE 预训练**：已经学会时间/空间位置的相对关系
3. **Action-effect 知识**：World Model 训练让它隐式学到了给定 action 画面会怎么变，微调到轨迹输出时这些知识可以迁移

---

## 5. 蒸馏训练细节

### 5.1 训练数据

```python
# 每条样本:
sample = {
    "image":        前视相机 (256×256),                    # → Encoder
    "ego_state":    [x, y, z, v, yaw, steer, ...],        # → State Encoder
    "goal":         [nav_x, nav_y],                        # → Goal Encoder
    "instruction":  "在前方路口左转",                      # → Lang Encoder
    
    # 标签 — 来自 CEM 规划结果:
    "trajectory":   [[x₁,y₁], [x₂,y₂], ..., [x₂₀,y₂₀]],  # 20 waypoints
}
```

### 5.2 Loss 设计

```python
# 主 loss: 模仿 CEM 专家
loss_traj = L1(pred_waypoints, expert_waypoints)

# 辅助 loss: 保证轨迹合理
loss_smooth = mean(‖w_{i+1} - w_i‖²)           # 平滑性
loss_boundary = relu(off_road_distance)         # 道路边界

# 可选: 对原始 World Model 任务做辅助监督, 防止灾难性遗忘
loss_world = L1(predictor(latent, action) → next_latent, target_latent)

# 总 loss
loss = loss_traj + α·loss_smooth + β·loss_boundary + γ·loss_world
```

### 5.3 训练配置

| 参数 | 值 |
|------|-----|
| Encoder | 冻结 V-JEPA 2 ViT |
| Decoder backbone | 从 AC Predictor 权重初始化 |
| Decoder 可训练 | 全部层 + trajectory head |
| 优化器 | AdamW, lr=1e-4 |
| 数据量 | CEM 为 10K+ 场景各生成一条轨迹 |
| 迭代 | 蒸馏 → 部署 → 采集新场景 → CEM 重新规划 → 重新蒸馏 |

---

## 6. 与纯模仿学习的对比

```
纯模仿学习:
  Encoder (随机/ImageNet预训练) + Decoder (随机初始化)
  → 在人类驾驶数据上训练
  → 学到: "人类一般怎么开"
  → 问题: OOD 场景容易崩，不会对物理做推理

本方案:
  Encoder (V-JEPA 2预训练) + Decoder (World Model预训练)
  → 先用 CEM 在 latent 空间为海量场景规划
  → 再用规划结果蒸馏
  → 学到: "理解 scene 动力学 + CEM 怎么规划"

差异:
  纯模仿复制行为 → 没见过的场景可能乱输出
  本方案内化了 World Model 的动力学理解和 CEM 的搜索能力
  → 虽然蒸馏后是 feed-forward，但 backbone 的参数里有动力学知识
```

### World Model 预训练到底贡献了什么

```
从 World Model 初始化 Decoder 时，backbone 的参数里已经隐式编码了:
  - "在这个场景里，给定 steer=0.1, throttle=0.3，车会往右偏"
  - "前方 10 米有障碍物的话，2 秒后会出现在视野中心"
  - "路口的结构意味着向左转的轨迹应该怎么弯曲"

虽然 Decoder 被训练成直接输出轨迹，但这些动力学知识不会完全丢失，
它变成了对轨迹预测的约束。
```

---

## 7. 完整 Pipeline 总览

```
┌──────────────────────────────────────────────────────────────────┐
│                       云端 (离线)                                  │
│                                                                    │
│  大规模场景库                                                      │
│       │                                                           │
│       ├──→ 1. World Model 训练                                    │
│       │       AC Predictor: (latent, action) → next_latent        │
│       │                                                           │
│       ├──→ 2. CEM 规划                                            │
│       │       每个场景搜索最优轨迹                                 │
│       │       → 形成 "场景→专家轨迹" 数据集                       │
│       │                                                           │
│       └──→ 3. 蒸馏训练                                            │
│               AC Predictor backbone → Trajectory Decoder          │
│               Loss: 模仿 CEM 专家 + 平滑 + 边界约束              │
│                                                                    │
│  输出: 部署权重 + 蒸馏模型                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       车载 (在线)                                  │
│                                                                    │
│  传感器输入:                                                       │
│    图像 + 导航 + 指令 + 自车状态                                   │
│       │                                                           │
│       ▼                                                           │
│  Encoder (冻结) → visual latent                                   │
│       │                                                           │
│       ▼                                                           │
│  Trajectory Decoder (回放 forward)                                │
│       │                                                           │
│       ▼                                                           │
│  waypoints [20, 2] → 控制器 → steer, throttle                    │
│                                                                    │
│  单帧耗时: ~20ms ✓                                                │
│  频率: 可达 50Hz                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| CEM 规划质量不足 | 蒸馏出来的 policy 学不到好行为 | 人工标注轨迹做混合监督；用 RL fine-tune 弥补 |
| World Model 在边缘场景推演不准 | CEM 规划的轨迹不合理 | 用在线数据持续更新 World Model；在安全场景验证 |
| 蒸馏后泛化退化 | 没见过的新场景表现差 | 定期采集新场景 → CEM 重新规划 → 重新蒸馏 |
| 灾难性遗忘 | Decoder 丢失 World Model 的动力学知识 | 辅助 loss 保留 world model 任务 |
| 安全关键场景不可靠 | 可能输出危险轨迹 | 规则安全层兜底；CEM 在线做安全验证 |

---

## 9. 迭代闭环

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. 云端: World Model + CEM 规划                        │
│       ↓                                                 │
│  2. 蒸馏: Decoder 学习专家轨迹                          │
│       ↓                                                 │
│  3. 部署: 车载推理 <20ms                                │
│       ↓                                                 │
│  4. 采集: 边缘场景 / 接管事件数据                       │
│       ↓                                                 │
│  5. 回灌: 新数据 → CEM 重新规划 → 加入训练集            │
│       ↓                                                 │
│  → 回到步骤 1                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 10. 总结一句话

**用 CEM + World Model 在云端做搜索，把搜索结果蒸馏到一个 feed-forward Decoder 里部署到车上 —— 在线不搜索，离线搜得狠。**
