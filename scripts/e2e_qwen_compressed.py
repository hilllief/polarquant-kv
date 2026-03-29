"""端到端测试：Qwen2.5-0.5B + PolarQuant-KV 压缩。"""

import torch
import time
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
from polarquant_kv_cuda.hf_integration import (
    patch_model, create_compressed_cache,
)

MODEL_PATH = "models/qwen2.5-0.5b"
DEVICE = "cuda"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"加载模型: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()

print(f"模型: {model.config.num_hidden_layers} 层, "
      f"{model.config.hidden_size} dim, "
      f"{model.config.num_attention_heads} Q heads, "
      f"{model.config.num_key_value_heads} KV heads, "
      f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

# Patch 模型
patch_model(model, n_bits=4, group_size=32)

prompts = [
    "请用三句话解释什么是KV Cache。",
    "Write a Python function to check if a number is prime.",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    # --- 标准推理 ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_std = model.generate(
            **inputs, max_new_tokens=80, do_sample=False, use_cache=True,
        )
    torch.cuda.synchronize()
    t_std = time.perf_counter() - t0
    text_std = tokenizer.decode(out_std[0][input_len:], skip_special_tokens=True)
    n_std = out_std.shape[1] - input_len

    # --- 压缩推理 ---
    cache = create_compressed_cache(model)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_comp = model.generate(
            **inputs, max_new_tokens=80, do_sample=False,
            use_cache=True, past_key_values=cache,
        )
    torch.cuda.synchronize()
    t_comp = time.perf_counter() - t0
    text_comp = tokenizer.decode(out_comp[0][input_len:], skip_special_tokens=True)
    n_comp = out_comp.shape[1] - input_len

    # 显存对比
    std_mem = cache.get_standard_memory_bytes()
    comp_mem = cache.get_compressed_memory_bytes()
    mem_save = (1 - comp_mem / std_mem) * 100 if std_mem > 0 else 0

    # Token 匹配
    min_len = min(out_std.shape[1], out_comp.shape[1])
    match = (out_std[0, :min_len] == out_comp[0, :min_len]).float().mean().item() * 100

    print(f"\n  标准: {n_std} tokens, {t_std:.2f}s, {n_std/t_std:.0f} tok/s")
    print(f"  压缩: {n_comp} tokens, {t_comp:.2f}s, {n_comp/t_comp:.0f} tok/s")
    print(f"  Token 匹配率: {match:.0f}%")
    print(f"  KV Cache: 标准 {std_mem/1024:.1f}KB → 压缩 {comp_mem/1024:.1f}KB, 节省 {mem_save:.0f}%")
    print(f"\n  标准输出: {text_std[:150]}")
    print(f"  压缩输出: {text_comp[:150]}")
