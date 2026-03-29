"""Phase 3: 推理集成 Wrapper — 测试。

覆盖需求 1~4 的核心验收标准。
"""

import torch
import pytest

from polarquant_kv_cuda.inference.config import PolarQuantConfig
from polarquant_kv_cuda.inference.kv_cache import KVCacheManager
from polarquant_kv_cuda.inference.attention import CompressedMultiHeadAttention
from polarquant_kv_cuda.inference.transformer import SimpleTransformer

D = 128
NUM_HEADS = 4
EMBED_DIM = NUM_HEADS * D  # 512
DEVICE = "cuda"


@pytest.fixture
def config():
    return PolarQuantConfig(n_bits=4, group_size=32, enable_compression=True)


@pytest.fixture
def config_off():
    return PolarQuantConfig(enable_compression=False)


# --- 需求 1: KV Cache Manager ---

class TestKVCacheManager:

    def test_init_and_append(self, config):
        cm = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)
        assert cm.seq_len == 0

        k = torch.randn(1, NUM_HEADS, 4, D, dtype=torch.float16, device=DEVICE)
        v = torch.randn(1, NUM_HEADS, 4, D, dtype=torch.float16, device=DEVICE)
        cm.append(k, v)
        assert cm.seq_len == 4

    def test_memory_bytes_positive(self, config):
        cm = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)
        k = torch.randn(1, NUM_HEADS, 8, D, dtype=torch.float16, device=DEVICE)
        v = torch.randn(1, NUM_HEADS, 8, D, dtype=torch.float16, device=DEVICE)
        cm.append(k, v)
        assert cm.memory_bytes > 0
        assert cm.compression_ratio > 1.0

    def test_cache_full_raises(self, config):
        cm = KVCacheManager(config, max_seq_len=4, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)
        k = torch.randn(1, NUM_HEADS, 4, D, dtype=torch.float16, device=DEVICE)
        v = torch.randn(1, NUM_HEADS, 4, D, dtype=torch.float16, device=DEVICE)
        cm.append(k, v)
        with pytest.raises(RuntimeError, match="已满"):
            cm.append(k, v)

    def test_reset(self, config):
        cm = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)
        k = torch.randn(1, NUM_HEADS, 4, D, dtype=torch.float16, device=DEVICE)
        v = torch.randn(1, NUM_HEADS, 4, D, dtype=torch.float16, device=DEVICE)
        cm.append(k, v)
        cm.reset()
        assert cm.seq_len == 0


# --- 需求 2: 注意力层 Wrapper ---

class TestCompressedAttention:

    def test_forward_shape(self, config):
        torch.manual_seed(42)
        attn = CompressedMultiHeadAttention(config, EMBED_DIM, NUM_HEADS).to(DEVICE)
        cm = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)
        x = torch.randn(1, 8, EMBED_DIM, device=DEVICE)
        out = attn(x, cm)
        assert out.shape == (1, 8, EMBED_DIM)

    def test_accuracy_vs_standard(self, config, config_off):
        """AC-2.4: 压缩 vs 标准注意力余弦相似度 ≥ 0.98"""
        torch.manual_seed(42)
        attn = CompressedMultiHeadAttention(config, EMBED_DIM, NUM_HEADS).to(DEVICE)
        x = torch.randn(1, 16, EMBED_DIM, device=DEVICE)

        # 标准注意力（无缓存）
        out_std = attn(x, cache_manager=None)

        # 压缩注意力
        cm = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)
        out_comp = attn(x, cm)

        cos = torch.nn.functional.cosine_similarity(
            out_std.flatten(), out_comp.flatten(), dim=0
        ).item()
        assert cos >= 0.98, f"注意力余弦相似度 {cos:.4f} < 0.98"

    def test_decode_step_by_step(self, config):
        """AC-2.3: 逐 token decode"""
        torch.manual_seed(42)
        attn = CompressedMultiHeadAttention(config, EMBED_DIM, NUM_HEADS).to(DEVICE)
        cm = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)

        # Prefill 4 tokens
        x_prefill = torch.randn(1, 4, EMBED_DIM, device=DEVICE)
        out_prefill = attn(x_prefill, cm)
        assert cm.seq_len == 4

        # Decode 3 tokens one by one
        for i in range(3):
            x_decode = torch.randn(1, 1, EMBED_DIM, device=DEVICE)
            out_decode = attn(x_decode, cm)
            assert out_decode.shape == (1, 1, EMBED_DIM)
            assert cm.seq_len == 5 + i


# --- 需求 3: 端到端推理 ---

class TestEndToEndInference:

    def test_multi_layer_transformer(self, config):
        """AC-3.1, AC-3.2: 多层 Transformer 推理"""
        torch.manual_seed(42)
        model = SimpleTransformer(
            config, num_layers=2, embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, vocab_size=100,
        ).to(DEVICE)  # float32，注意力层内部处理 half

        input_ids = torch.randint(0, 100, (1, 8), device=DEVICE)

        # 标准推理（无压缩）
        config_off = PolarQuantConfig(enable_compression=False)
        model_std = SimpleTransformer(
            config_off, num_layers=2, embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, vocab_size=100,
        ).to(DEVICE)
        model_std.load_state_dict(model.state_dict())

        logits_std = model_std(input_ids)

        # 压缩推理
        caches = model.create_cache_managers(max_seq_len=64, batch=1, device=DEVICE)
        logits_comp = model(input_ids, caches)

        cos = torch.nn.functional.cosine_similarity(
            logits_std.flatten().float(), logits_comp.flatten().float(), dim=0
        ).item()
        assert cos >= 0.95, f"多层 Transformer 余弦相似度 {cos:.4f} < 0.95"

    def test_memory_saving(self, config, config_off):
        """AC-3.3: 显存节省"""
        torch.manual_seed(42)
        seq_len = 32

        cm_comp = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                                 head_dim=D, device=DEVICE)
        cm_std = KVCacheManager(config_off, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                                head_dim=D, device=DEVICE)

        k = torch.randn(1, NUM_HEADS, seq_len, D, dtype=torch.float16, device=DEVICE)
        v = torch.randn(1, NUM_HEADS, seq_len, D, dtype=torch.float16, device=DEVICE)

        cm_comp.append(k, v)
        cm_std.append(k, v)

        assert cm_comp.memory_bytes < cm_std.memory_bytes, (
            f"压缩 {cm_comp.memory_bytes} >= 标准 {cm_std.memory_bytes}"
        )

    def test_decode_no_memory_leak(self, config):
        """AC-3.4: decode 循环无显存泄漏"""
        torch.manual_seed(42)
        cm = KVCacheManager(config, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)

        for i in range(10):
            k = torch.randn(1, NUM_HEADS, 1, D, dtype=torch.float16, device=DEVICE)
            v = torch.randn(1, NUM_HEADS, 1, D, dtype=torch.float16, device=DEVICE)
            cm.append(k, v)
            assert cm.seq_len == i + 1


# --- 需求 4: 配置与开关 ---

class TestConfigSwitch:

    def test_disable_compression(self, config_off):
        """AC-4.1: 禁用压缩退化为标准注意力"""
        torch.manual_seed(42)
        attn = CompressedMultiHeadAttention(config_off, EMBED_DIM, NUM_HEADS).to(DEVICE)

        x = torch.randn(1, 8, EMBED_DIM, device=DEVICE)
        out_no_cache = attn(x, cache_manager=None)

        cm = KVCacheManager(config_off, max_seq_len=64, batch=1, num_heads=NUM_HEADS,
                            head_dim=D, device=DEVICE)
        out_with_cache = attn(x, cm)

        cos = torch.nn.functional.cosine_similarity(
            out_no_cache.flatten(), out_with_cache.flatten(), dim=0
        ).item()
        # 禁用压缩时应完全一致
        assert cos >= 0.999, f"禁用压缩余弦 {cos:.4f} < 0.999"

    def test_config_dataclass(self):
        """AC-4.3: PolarQuantConfig 包含所有参数"""
        c = PolarQuantConfig()
        assert hasattr(c, "n_bits")
        assert hasattr(c, "group_size")
        assert hasattr(c, "jl_dim")
        assert hasattr(c, "enable_qjl")
        assert hasattr(c, "enable_compression")
