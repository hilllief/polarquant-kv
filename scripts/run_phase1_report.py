"""生成 Phase 1 完成判定报告。"""

import logging
logging.basicConfig(level=logging.INFO)

from polarquant_kv.benchmark import hyperparameter_search, generate_phase1_report

results = hyperparameter_search(
    n_bits_range=[2, 3, 4, 6, 8],
    group_size_range=[16, 32, 64, 128],
    jl_dim_range=[32, 64, 128],
    d=128,
    num_samples=50,
)

report = generate_phase1_report(results)
print(report)
print()
print("--- 全部配置详情（按压缩比降序）---")
for c in sorted(results["configs"], key=lambda x: -x["compression_ratio"]):
    nb = c["n_bits"]
    gs = c["group_size"]
    jl = c["jl_dim"]
    cr = c["compression_ratio"]
    cs = c["cosine_similarity"]
    ac = c["attention_output_cosine_similarity"]
    print(f"  n_bits={nb}, gs={gs}, jl={jl}: ratio={cr:.2f}x, cos={cs:.4f}, attn_cos={ac:.4f}")

# 保存报告
with open("docs/phase1-report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("\n报告已保存到 docs/phase1-report.md")
