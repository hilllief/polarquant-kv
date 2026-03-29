"""端到端测试：用真实 Qwen2.5-0.5B 模型验证 PolarQuant-KV。"""

import torch
import time
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/qwen2.5-0.5b"
DEVICE = "cuda"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"加载模型: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()

print(f"模型加载完成: {model.config.num_hidden_layers} 层, "
      f"{model.config.hidden_size} dim, "
      f"{model.config.num_attention_heads} Q heads, "
      f"{model.config.num_key_value_heads} KV heads")

# 测试 prompt
prompts = [
    "请用三句话解释什么是KV Cache。",
    "Write a Python function to check if a number is prime.",
    "What are the main causes of climate change?",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    # 标准推理
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    mem_after = torch.cuda.max_memory_allocated()
    kv_mem = mem_after - mem_before

    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    n_tokens = outputs.shape[1] - input_len
    speed = n_tokens / (t1 - t0)

    print(f"  生成 {n_tokens} tokens, {t1-t0:.2f}s, {speed:.0f} tok/s")
    print(f"  峰值显存增量: {kv_mem/1e6:.1f} MB")
    print(f"  输出: {generated[:200]}")

print(f"\n{'='*60}")
print("模型信息（用于 PolarQuant-KV 集成）:")
print(f"  head_dim = {model.config.hidden_size // model.config.num_attention_heads}")
print(f"  num_layers = {model.config.num_hidden_layers}")
print(f"  num_q_heads = {model.config.num_attention_heads}")
print(f"  num_kv_heads = {model.config.num_key_value_heads}")
print(f"  GQA ratio = {model.config.num_attention_heads // model.config.num_key_value_heads}:1")
