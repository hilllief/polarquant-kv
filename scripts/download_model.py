"""用国内镜像下载 Qwen2.5-0.5B 模型。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("开始下载 Qwen2.5-0.5B（从 hf-mirror.com）...")
snapshot_download(
    "Qwen/Qwen2.5-0.5B",
    local_dir="models/qwen2.5-0.5b",
)
print("下载完成！")
