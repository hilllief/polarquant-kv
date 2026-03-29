"""端到端测试：用真实 Transformer 架构验证 PolarQuant-KV。

使用 Qwen2.5 的架构（随机权重），验证：
1. KV Cache 压缩后推理是否正常
2. 显存节省多少
3. 速度对比
4. 生成质量对比（token 匹配率）
"""

import torch
import torch.nn as nn
import math
import time

from polarquant_kv_cuda.inference.config import PolarQuantConfig
from polarquant_kv_cuda.inference.kv_cache import KVCacheManager
from polarquant_kv_cuda.inference.transformer import SimpleTransformer

DEVICE = "cuda"
torch.manual_seed(42)


def build_model(config, num_layers=8, embed_dim=512, num_heads=8,
                num_kv_heads=4, vocab_size=32000):
    """构建一个类 Qwen 架构的模型（GQA + 多层）。"""
    model = SimpleTransformer(
        config=config,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
    ).to(DEVICE)
    return model


def generate(model, input_ids, max_new_tokens=50, use_cache=True):
    """自回归生成。"""
    config = model.config
    B = input_ids.shape[0]

    if use_cache:
        caches = model.create_cache_managers(
            max_seq_len=input_ids.shape[1] + max_new_tokens,
            batch=B, device=DEVICE,
        )
    else:
        caches = None

    # Prefill
    with torch.no_grad():
        logits = model(input_ids, caches)
    next_token = logits[:, -1:, :].argmax(dim=-1)
    generated = [next_token]

    # Decode
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            logits = model(next_token, caches)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token)

    return torch.cat(generated, dim=1), caches


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    NUM_LAYERS = 8
    EMBED_DIM = 512
    NUM_HEADS = 8
    NUM_KV_HEADS = 4  # GQA: 4 KV heads, 8 Q heads
    VOCAB_SIZE = 32000
    MAX_NEW = 50

    # 构建两个相同权重的模型
    config_std = PolarQuantConfig(enable_compression=False)
    config_comp = PolarQuantConfig(n_bits=4, group_size=32, enable_compression=True)

    model_std = build_model(config_std, NUM_LAYERS, EMBED_DIM, NUM_HEADS, NUM_KV_HEADS, VOCAB_SIZE)
    model_comp = build_model(config_comp, NUM_LAYERS, EMBED_DIM, NUM_HEADS, NUM_KV_HEADS, VOCAB_SIZE)
    model_comp.load_state_dict(model_std.state_dict())

    print(f"模型: {NUM_LAYERS} 层, {EMBED_DIM} dim, {NUM_HEADS} Q heads, {NUM_KV_HEADS} KV heads")
    print(f"词表: {VOCAB_SIZE}, GQA ratio: {NUM_HEADS // NUM_KV_HEADS}:1\n")

    # 测试不同 prompt 长度
    for prompt_len in [32, 128, 512]:
        print(f"{'='*60}")
        print(f"Prompt 长度: {prompt_len} tokens, 生成: {MAX_NEW} tokens")
        print(f"{'='*60}")

        input_ids = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=DEVICE)

        # 标准推理
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tokens_std, caches_std = generate(model_std, input_ids, MAX_NEW, use_cache=True)
        torch.cuda.synchronize()
        t_std = time.perf_counter() - t0
        mem_std = sum(c.memory_bytes for c in caches_std) if caches_std else 0

        # 压缩推理
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens_comp, caches_comp = generate(model_comp, input_ids, MAX_NEW, use_cache=True)
        torch.cuda.synchronize()
        t_comp = time.perf_counter() - t0
        mem_comp = sum(c.memory_bytes for c in caches_comp) if caches_comp else 0

        # Token 匹配率
        match = (tokens_std == tokens_comp).float().mean().item() * 100

        # 显存节省
        mem_save = (1 - mem_comp / mem_std) * 100 if mem_std > 0 else 0

        total_tokens = prompt_len + MAX_NEW
        speed_std = total_tokens / t_std
        speed_comp = total_tokens / t_comp

        print(f"  标准推理: {t_std:.2f}s, {speed_std:.0f} tok/s, KV Cache {mem_std/1024:.1f} KB")
        print(f"  压缩推理: {t_comp:.2f}s, {speed_comp:.0f} tok/s, KV Cache {mem_comp/1024:.1f} KB")
        print(f"  Token 匹配率: {match:.0f}%")
        print(f"  显存节省: {mem_save:.0f}%")
        print(f"  速度比: {speed_comp/speed_std:.2f}x")
        print(f"  生成的 token (标准): {tokens_std[0, :10].tolist()}...")
        print(f"  生成的 token (压缩): {tokens_comp[0, :10].tolist()}...")
        print()

        del caches_std, caches_comp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
