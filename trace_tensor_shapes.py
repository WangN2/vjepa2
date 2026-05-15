"""
张量形状追踪：V-JEPA 2 ViT 前向传播
用随机权重构建 ViT-L，追踪每个步骤的 tensor 形状变化
"""

import torch
from src.models.vision_transformer import VisionTransformer, vit_large


def trace_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")

    # ============================================================
    # Step 0: 构造虚拟视频输入
    # ============================================================
    # V-JEPA 默认: 16帧, 224x224, 4fps
    B, C, T, H, W = 1, 3, 16, 224, 224
    video = torch.randn(B, C, T, H, W).to(device)

    print("=" * 60)
    print("Step 0: 输入视频")
    print("=" * 60)
    print(f"形状: {list(video.shape)}")
    print(f"含义: batch={B}, 通道={C}(RGB), 帧数={T}, 高={H}, 宽={W}")
    print()

    # ============================================================
    # Step 1: 构建 ViT-L
    # ============================================================
    # 默认 ViT-L 是图片版（num_frames=1），这里要指定视频参数
    model = VisionTransformer(
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,       # 视频输入: 16帧
        tubelet_size=2,      # 每2帧合并为一个时间步
        in_chans=3,
        embed_dim=1024,      # ViT-L 的工作维度
        depth=24,            # 24 层 Transformer
        num_heads=16,        # 16 头注意力
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_silu=False,       # V-JEPA 默认用 GELU (非 SiLU)
        wide_silu=True,
        use_sdpa=True,
        use_rope=False,       # 先用标准 pos_embed 方便理解
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ViT-L 参数量: {total_params/1e6:.1f}M")
    print()

    # ============================================================
    # Step 2: PatchEmbed3D (内部在 forward 中调用)
    # ============================================================
    print("=" * 60)
    print("Step 1: PatchEmbed3D — 视频切分 patch")
    print("=" * 60)

    # 手动调用 patch_embed 来看形状
    x = model.patch_embed(video)
    print(f"PatchEmbed3D 输出: {list(x.shape)}")
    print(f"  说明:")
    print(f"    - 时间步: T/tubelet_size = 16/2 = 8")
    print(f"    - 空间: H/patch_size × W/patch_size = 14 × 14 = 196")
    print(f"    - 总 tokens: 8 × 196 = {x.shape[1]}")
    print(f"    - 每个 token 维度: {x.shape[-1]}")
    print()

    # ============================================================
    # Step 3: 位置编码
    # ============================================================
    print("=" * 60)
    print("Step 2: 加位置编码")
    print("=" * 60)

    # V-JEPA 不适用 cls_token（与 ViT 原版不同！）
    # 直接加 sincos 位置编码
    print(f"位置编码形状: {list(model.pos_embed.shape)}")
    print(f"  含义: (1, {model.pos_embed.shape[1]} 个位置, {model.pos_embed.shape[-1]} 维)")

    # 加位置编码
    x = x + model.pos_embed
    print(f"加位置编码后: {list(x.shape)}")
    print()

    # ============================================================
    # Step 4: Transformer Blocks (24 层)
    # ============================================================
    n_layers = len(model.blocks)
    print("=" * 60)
    print(f"Step 3: {n_layers} 层 Transformer Block")
    print("=" * 60)
    print(f"每个 Block: LN → Multi-Head Self-Attention → Residual → LN → MLP → Residual")
    print()

    for i, blk in enumerate(model.blocks):
        x = blk(x)
        if i == 0:
            print(f"  第 1 层  输出: {list(x.shape)}")
        elif i == n_layers // 2 - 1:
            print(f"  第 12 层 输出: {list(x.shape)} (中间层)")
        elif i == n_layers - 1:
            print(f"  第 24 层 输出: {list(x.shape)} (最后一层)")
    print()

    # ============================================================
    # Step 5: 最终输出
    # ============================================================
    print("=" * 60)
    print("Step 4: 最终输出")
    print("=" * 60)

    x = model.norm(x)
    print(f"LayerNorm 后: {list(x.shape)}")
    print()

    # ============================================================
    # 总结
    # ============================================================
    print("=" * 60)
    print("前向传播总结")
    print("=" * 60)
    print(f"输入:  {list(video.shape)}")
    print(f"       ┃")
    print(f"  ┏━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print(f"  ┃  PatchEmbed3D (Conv3D, kernel=2×16×16) ┃")
    print(f"  ┃  16帧 ÷ 2(tubelet) = 8 时间步         ┃")
    print(f"  ┃  224 ÷ 16(patch) = 14 → 14×14=196     ┃")
    print(f"  ┃  8 × 196 = {model.num_patches} tokens              ┃")
    print(f"  ┗━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print(f"       ┃")
    print(f"  ┏━━━┻━━━━━━━━━━━━━━━━━━━━┓")
    print(f"  ┃  + sincos 位置编码     ┃")
    print(f"  ┗━━━┳━━━━━━━━━━━━━━━━━━━━┛")
    print(f"       ┃")
    print(f"  ┏━━━┻━━━━━━━━━━━━━━┓")
    print(f"  ┃  {n_layers}× Transformer Block  ┃")
    print(f"  ┃  token 数不变: {model.num_patches}     ┃")
    print(f"  ┃  维度不变: {model.embed_dim}          ┃")
    print(f"  ┗━━━┳━━━━━━━━━━━━━━┛")
    print(f"       ┃")
    print(f"  ┏━━━┻━━━━┓")
    print(f"  ┃  LayerNorm ┃")
    print(f"  ┗━━━┳━━━━┛")
    print(f"       ┃")
    print(f"输出:  {list(x.shape)}")
    print()
    print(f"模型总参数量: {total_params/1e6:.1f}M")
    print(f"注意: V-JEPA 不使用 cls_token，所有 {model.num_patches} 个 patch tokens 都传给 Predictor")


if __name__ == "__main__":
    trace_shapes()
