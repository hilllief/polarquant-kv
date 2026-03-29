"""PolarQuant-KV 效果演示。

展示：压缩精度、显存节省、prefill + decode 推理流程。
"""

import torch
import torch.nn.functional as F

from polarquant_kv_cuda.inference.config import PolarQuantConfig
from polarquant_kv_cuda.inference.kv_cache import KVCacheManager
from polarquant_kv_cuda.inference.transformer import SimpleTransformer

DEVICE = "cuda"
torch.manual_seed(42)


def demo_compression_accuracy():
    """演示不同 bit 数下的压缩精度。"""
    print("=" * 60)
    print("1. 压缩精度对比")
    print("=" * 60)

    from polarquant_kv_cuda.rotation import generate_rotation_matrix
    from polarquant_kv_cuda.compress_kernel import compress_gpu
    from polarquant_kv_cuda.decompress_kernel import decompress_gpu

    D = 128
    R = generate_rotation_matrix(D, seed=42, device=DEVICE)
    kv = torch.randn(1, 1, 100, D, dtype=torch.float16, device=DEVICE)

    print(f"\n  原始 KV: shape={list(kv.shape)}, dtype={kv.dtype}")
    print(f"  原始显存: {kv.nelement() * 2 / 1024:.1f} KB\n")

    for n_bits in [2, 3, 4, 6, 8]:
        compressed = compress_gpu(kv, R, n_bits=n_bits, group_size=32)
        kv_hat = decompress_gpu(compressed, R)

        cos = F.cosine_similarity(
            kv.reshape(-1, D).float(), kv_hat.reshape(-1, D).float(), dim=1
        ).mean().item()

        from polarquant_kv_cuda.compressor import get_memory_bytes
        comp_bytes = get_memory_bytes(compressed)
        orig_bytes = kv.nelement() * 2
        ratio = orig_bytes / comp_bytes

        print(f"  {n_bits}-bit: 余弦相似度={cos:.4f}, 压缩比={ratio:.2f}x, "
              f"显存 {orig_bytes/1024:.1f}KB → {comp_bytes/1024:.1f}KB")


def demo_memory_saving():
    """演示长序列下的显存节省。"""
    print("\n" + "=" * 60)
    print("2. 长序列显存节省")
    print("=" * 60)

    config = PolarQuantConfig(n_bits=4, group_size=32)
    config_off = PolarQuantConfig(enable_compression=False)
    num_heads = 32
    head_dim = 128

    print(f"\n  配置: {num_heads} heads × {head_dim} dim, 4-bit 量化\n")
    print(f"  {'seq_len':>8} | {'标准 KV':>10} | {'压缩 KV':>10} | {'节省':>6} | {'压缩比':>6}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*6} | {'-'*6}")

    for seq_len in [128, 512, 2048, 4096, 8192]:
        cm_std = KVCacheManager(config_off, seq_len, 1, num_heads, head_dim, DEVICE)
        cm_comp = KVCacheManager(config, seq_len, 1, num_heads, head_dim, DEVICE)

        k = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE)
        v = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE)

        cm_std.append(k, v)
        cm_comp.append(k, v)

        std_mb = cm_std.memory_bytes / 1e6
        comp_mb = cm_comp.memory_bytes / 1e6
        saving = (1 - comp_mb / std_mb) * 100
        ratio = std_mb / comp_mb

        print(f"  {seq_len:>8} | {std_mb:>8.1f}MB | {comp_mb:>8.1f}MB | {saving:>4.0f}% | {ratio:>5.2f}x")

        del k, v, cm_std, cm_comp
        torch.cuda.empty_cache()


def demo_inference():
    """演示完整的 prefill + decode 推理流程。"""
    print("\n" + "=" * 60)
    print("3. Transformer 推理演示 (prefill + decode)")
    print("=" * 60)

    embed_dim = 512
    num_heads = 4
    num_layers = 2
    vocab_size = 1000
    max_seq = 64

    # 创建模型
    config_comp = PolarQuantConfig(n_bits=4, group_size=32)
    config_std = PolarQuantConfig(enable_compression=False)

    model = SimpleTransformer(
        config_comp, num_layers, embed_dim, num_heads, vocab_size=vocab_size
    ).to(DEVICE)

    model_std = SimpleTransformer(
        config_std, num_layers, embed_dim, num_heads, vocab_size=vocab_size
    ).to(DEVICE)
    model_std.load_state_dict(model.state_dict())

    # Prefill
    prompt = torch.randint(0, vocab_size, (1, 8), device=DEVICE)
    print(f"\n  Prompt tokens: {prompt[0].tolist()}")

    caches_comp = model.create_cache_managers(max_seq, 1, DEVICE)
    caches_std = model_std.create_cache_managers(max_seq, 1, DEVICE)

    logits_comp = model(prompt, caches_comp)
    logits_std = model_std(prompt, caches_std)

    cos = F.cosine_similarity(logits_comp.flatten(), logits_std.flatten(), dim=0).item()
    print(f"  Prefill logits 余弦相似度: {cos:.4f}")

    # Decode 10 tokens
    print(f"\n  Decode 生成 10 tokens:")
    generated_comp = []
    generated_std = []
    next_comp = logits_comp[:, -1:, :].argmax(dim=-1)
    next_std = logits_std[:, -1:, :].argmax(dim=-1)

    for step in range(10):
        logits_comp = model(next_comp, caches_comp)
        logits_std = model_std(next_std, caches_std)

        next_comp = logits_comp[:, -1:, :].argmax(dim=-1)
        next_std = logits_std[:, -1:, :].argmax(dim=-1)

        generated_comp.append(next_comp.item())
        generated_std.append(next_std.item())

    print(f"    标准: {generated_std}")
    print(f"    压缩: {generated_comp}")
    match = sum(a == b for a, b in zip(generated_std, generated_comp))
    print(f"    token 匹配率: {match}/10")

    # 显存对比
    std_mem = sum(c.memory_bytes for c in caches_std)
    comp_mem = sum(c.memory_bytes for c in caches_comp)
    print(f"\n  KV Cache 显存:")
    print(f"    标准: {std_mem/1024:.1f} KB")
    print(f"    压缩: {comp_mem/1024:.1f} KB")
    print(f"    节省: {(1 - comp_mem/std_mem)*100:.0f}%")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    demo_compression_accuracy()
    demo_memory_saving()
    demo_inference()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
