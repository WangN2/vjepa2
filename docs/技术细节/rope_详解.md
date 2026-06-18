# RoPE（旋转位置编码）详解

> 从 2D 旋转 → 3D 视频 RoPE → V-JEPA 2 源码逐行拆解

---

## 一、为什么需要位置编码

Self-Attention 是**排列不变**的——交换两个 token 的位置，Attention 输出不变。必须注入位置信息。

两种思路：
- **绝对位置编码**：给每个位置一个唯一的向量，加到 token 上
- **相对位置编码**：让 Attention 内积只依赖两个 token 的相对距离

RoPE 属于相对位置编码，通过**旋转向量**来注入位置。

---

## 二、2D 旋转公式

把一个向量 `(x, y)` 旋转角度 θ：

```
x' = x·cosθ - y·sinθ
y' = y·cosθ + x·sinθ
```

关键性质：两个向量分别旋转 θa 和 θb 后，内积 = 原始内积旋转 `(θa - θb)`。所以 **Q·K 只依赖位置差，不依赖绝对位置**。

---

## 三、高维旋转

D 维向量拆成 D/2 个 2D 对，每对用不同频率旋转：

```
64维: [x0,y0, x1,y1, x2,y2, ..., x31,y31]
        频率0    频率1    频率2        频率31
```

频率基底公式：`omega = 1.0 / 10000^(i / (D/2))`

- i=0 → 频率=1.0（高频，近处敏感）
- i=31 → 频率≈0.0001（低频，远处也有效）

---

## 四、源码：`rotate_queries_or_keys(x, pos)`

文件：[src/models/utils/modules.py:26-55](src/models/utils/modules.py#L26)

```python
def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()

    # 1. 计算频率基底 [D/2]
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega

    # 2. einsum 外积：每个 token 每个频率计算角度 θ
    freq = torch.einsum("..., f -> ... f", pos, omega)

    # 3. sin 和 cos
    emb_sin = freq.sin()
    emb_cos = freq.cos()

    # 4. repeat 复制成对（已知 bug：应用 repeat_interleave）
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # 5. 拆成 2D 对，旋转
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)   # (-y2, y1) = 旋转90度
    y = y.flatten(-2)

    # 6. 加权求和：x·cosθ + y_rotated·sinθ
    return (x * emb_cos) + (y * emb_sin)
```

---

## 五、3D 视频 RoPE

### 5.1 视频 → 3D 网格

```
16帧视频 ÷ tubelet_size=2 → 8 个时间步
224÷16=14 → 14×14 个空间位置
总 token 数 = 8 × 14 × 14 = 1568
```

每个 token 有三维坐标：(帧号, 行号, 列号)

### 5.2 head_dim 切分

```
64 维 head_dim (1024÷16=64):
|← 20维(depth) →|← 20维(height) →|← 20维(width) →|← 4维不动 →|
  10对 用帧号旋转   10对 用行号旋转   10对 用列号旋转    缓冲
```

### 5.3 `separate_positions` — 从 flat ID 还原坐标

```
帧号 = id ÷ (H_patches × W_patches)
行号 = (id - tokens_per_frame × 帧号) ÷ W_patches
列号 = (id - tokens_per_frame × 帧号) - tokens_per_row × 行号
```

### 5.4 `RoPEAttention.forward` 关键步骤

```python
# 1. 从 flat token ID 算出 3D 坐标
frame_ids, height_ids, width_ids = self.separate_positions(ids, H_patches, W_patches)

# 2. 三段分别用不同坐标旋转
qd = rotate_queries_or_keys(q[..., :20], pos=frame_ids)      # depth
qh = rotate_queries_or_keys(q[..., 20:40], pos=height_ids)  # height
qw = rotate_queries_or_keys(q[..., 40:60], pos=width_ids)   # width

# 3. 拼接：三段旋转后的 + 4 维不动的
q = cat([qd, qh, qw, q[..., 60:64]], dim=-1)
```

---

## 六、直觉总结

- RoPE = 在 Attention 计算前，用旋转给 Q 和 K 注入位置信息
- 2D 核心：拆成数对 → 每对旋转不同角度 → 内积只依赖相对位置
- 3D 扩展：把 head_dim 切成三段 → 每段用不同空间坐标旋转 → 模型同时感知三个维度的位置差异
- 优势：自然支持变长输入，训练 16 帧 → 推理可用 32 帧
