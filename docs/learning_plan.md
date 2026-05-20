# V-JEPA 2 学习计划（实时更新）

> 学习目标：从零 DL 基础 → 能理解、运行、修改 V-JEPA 2 代码
> 开始日期：2026-05-11 | 更新日期：2026-05-20

---

## ✅ 阶段一：DL 基础概念（已完成）

| 概念 | 完成情况 |
|------|---------|
| 表征学习（Representation Learning） | ✅ |
| 自监督学习（SSL） | ✅ |
| Vision Transformer（ViT）原理 | ✅ |
| EMA / 动量编码器 | ✅ |
| L1 vs L2 Loss | ✅ |
| RoPE 位置编码（概念） | ✅ |

---

## ✅ 阶段二：跑通完整流程（已完成）

- ✅ 环境搭建（conda + pip install -e .）
- ✅ 张量形状追踪（`trace_tensor_shapes.py`）
- ✅ 随机权重推理流程（`run_inference.py`）
- ✅ 遮罩策略详解（`multiseq_multiblock3d.py`）
- ✅ 训练循环数据流（`app/vjepa/train.py`）
- ✅ 预训练模型下载（vitl.pt, 4.78GB）
- ✅ 真实权重推理（`run_inference_pretrained.py`）

---

## ✅ 阶段三：应用 + 可视化（已完成）

- ✅ V-JEPA 能力全景理解
- ✅ 场景变化检测 Demo（`demo_app_scenes.png`）
- ✅ 相似场景检索 Demo（`demo_app_retrieval.png`）
- ✅ 帧序打乱检测 Demo（`demo_shuffle.png`）
- ✅ 世界模型预测 Demo（`demo_worldmodel.png`）
- ✅ 特征空间最近邻可视化（`demo_visualize.py`）
- ✅ 特征匹配可视化（`demo_similarity.py`）
- ✅ 场景变化预警 + 架构图（`demo_forecast.py`）
- ✅ 分类/微调演示（`demo_finetune.py`，AttentiveClassifier，96.6%）
- ✅ V2 vs V2.1 差异对比（dense features, ImageNet joint, context loss）
- ✅ V-JEPA 2-AC 动作条件预测架构（action tokens, causal attention）

---

## 阶段四：深入代码组件（当前进行中）

逐行读核心组件实现：

1. **RoPEAttention** — 旋转位置编码在 Attention 中的具体实现
   - 文件：`src/models/utils/modules.py` → `apply_rotary_emb_xyz`
   - 理解 3D 位置编码如何在 Q/K 上旋转
2. **SwiGLU FFN** — 门控线性单元变体
   - 文件：`src/models/utils/modules.py` → `SwiGLUFFN`
   - 理解为什么 SwiGLU 比标准 FFN 效果更好
3. **Attention 实现对比** — `RoPEAttention` vs `Attention` vs SDPA
4. **Loss 与 EMA 调度** — 细读 `app/vjepa/train.py` 中的 EMA 调度和 loss_fn

---

## 阶段五：动手实验（可选，需要 GPU）

1. 用 `configs/train_2_1/vitb16/` 训练小模型
2. 修改损失函数，在 V2 的 loss_fn 中尝试 L2
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
