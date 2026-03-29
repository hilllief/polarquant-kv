"""需求 5: 超参数搜索与精度验证 — 单元测试。

覆盖 AC-5.1 ~ AC-5.7。
"""

import logging
import math

import numpy as np
import pytest

from polarquant_kv.benchmark import (
    evaluate_config,
    hyperparameter_search,
    generate_phase1_report,
)
from polarquant_kv.utils import compute_compression_ratio


class TestCompressionRatioFormula:
    """AC-5.5: 压缩比公式验证"""

    @pytest.mark.parametrize(
        "d, n_bits, group_size, jl_dim",
        [
            (128, 4, 32, 64),
            (128, 3, 32, 64),
            (128, 2, 32, 64),
            (128, 8, 32, 64),
            (64, 4, 16, 32),
            (256, 4, 64, 128),
        ],
    )
    def test_formula_matches_definition(self, d, n_bits, group_size, jl_dim):
        """验证实现与 AC-5.5 定义的公式一致。"""
        num_groups = math.ceil(d / group_size)
        expected = (d * 16) / (16 + n_bits * d + num_groups * 32 + jl_dim)
        actual = compute_compression_ratio(d, n_bits, group_size, jl_dim)
        assert abs(actual - expected) < 1e-10, (
            f"公式不匹配: actual={actual:.6f}, expected={expected:.6f}"
        )


class TestStorageBytesAccuracy:
    """AC-5.2: 存储字节数精确计算"""

    def test_4bit_128d_storage(self):
        """手动计算 d=128, 4-bit, group_size=32, jl_dim=64 的存储。"""
        d, n_bits, group_size, jl_dim = 128, 4, 32, 64
        num_groups = math.ceil(d / group_size)  # 4

        # 按 bit 计算
        radius_bits = 16
        quant_bits = n_bits * d  # 512
        group_param_bits = num_groups * 32  # 128 (min + scale 各 FP16)
        jl_bits = jl_dim  # 64

        total_bits = radius_bits + quant_bits + group_param_bits + jl_bits
        original_bits = d * 16

        expected_ratio = original_bits / total_bits
        actual_ratio = compute_compression_ratio(d, n_bits, group_size, jl_dim)
        assert abs(actual_ratio - expected_ratio) < 1e-10


class TestEvaluateConfig:
    """AC-5.1, AC-5.3: 单配置评估"""

    def test_returns_required_metrics(self):
        result = evaluate_config(n_bits=4, group_size=32, jl_dim=64, d=128, num_samples=10, seed=42)
        assert "compression_ratio" in result
        assert "cosine_similarity" in result
        assert "attention_score_mse" in result
        assert "attention_output_cosine_similarity" in result

    def test_compression_ratio_positive(self):
        result = evaluate_config(n_bits=4, group_size=32, jl_dim=64, d=128, num_samples=10, seed=42)
        assert result["compression_ratio"] > 1.0

    def test_cosine_similarity_in_range(self):
        result = evaluate_config(n_bits=4, group_size=32, jl_dim=64, d=128, num_samples=10, seed=42)
        assert 0.0 <= result["cosine_similarity"] <= 1.0


class TestHyperparameterSearch:
    """AC-5.1: 超参数组合遍历"""

    def test_all_combinations_covered(self):
        n_bits_range = [4, 8]
        group_size_range = [32, 64]
        jl_dim_range = [64]

        results = hyperparameter_search(
            n_bits_range=n_bits_range,
            group_size_range=group_size_range,
            jl_dim_range=jl_dim_range,
            d=128,
            num_samples=5,
        )
        expected_count = len(n_bits_range) * len(group_size_range) * len(jl_dim_range)
        assert len(results["configs"]) == expected_count

    def test_each_config_has_metrics(self):
        results = hyperparameter_search(
            n_bits_range=[4],
            group_size_range=[32],
            jl_dim_range=[64],
            d=128,
            num_samples=5,
        )
        for config in results["configs"]:
            assert "n_bits" in config
            assert "group_size" in config
            assert "jl_dim" in config
            assert "compression_ratio" in config
            assert "cosine_similarity" in config


class TestParetoRecommendation:
    """AC-5.4: 帕累托最优推荐"""

    def test_recommendation_meets_criteria(self):
        results = hyperparameter_search(
            n_bits_range=[4, 6, 8],
            group_size_range=[32],
            jl_dim_range=[64],
            d=128,
            num_samples=20,
        )
        # 应该有推荐配置（可能是放宽条件后的）
        assert results.get("recommended") is not None
        rec = results["recommended"]
        assert rec["compression_ratio"] > 1.0
        assert rec["cosine_similarity"] > 0.9

    def test_no_recommendation_when_impossible(self):
        """2-bit 可能无法满足精度要求。"""
        results = hyperparameter_search(
            n_bits_range=[2],
            group_size_range=[128],
            jl_dim_range=[32],
            d=128,
            num_samples=20,
        )
        # 如果没有满足条件的配置，应有明确提示
        if results.get("recommended") is None:
            assert "no_recommendation_reason" in results


class TestProgressLogging:
    """AC-5.6: 进度日志"""

    def test_logs_progress(self, caplog):
        with caplog.at_level(logging.INFO):
            hyperparameter_search(
                n_bits_range=[4, 8],
                group_size_range=[32],
                jl_dim_range=[64],
                d=128,
                num_samples=5,
            )
        # 应该有进度日志
        progress_logs = [r for r in caplog.records if "进度" in r.message or "progress" in r.message.lower()]
        assert len(progress_logs) >= 2, "缺少进度日志输出"


class TestPhase1Report:
    """AC-5.7: Phase 1 完成判定报告"""

    def test_report_contains_required_sections(self):
        results = hyperparameter_search(
            n_bits_range=[4],
            group_size_range=[32],
            jl_dim_range=[64],
            d=128,
            num_samples=20,
        )
        report = generate_phase1_report(results)
        assert isinstance(report, str)
        assert len(report) > 0
        # 报告应包含三个判定项
        assert "压缩比" in report or "compression" in report.lower()
        assert "QJL" in report or "修正" in report
        assert "推荐" in report or "recommend" in report.lower()
