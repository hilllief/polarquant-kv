# PolarQuant-KV 创新优化方案

## 发现的关键差距

我们的实现 vs TurboQuant 论文的核心差异：

```
我们当前的实现（per-group min/max 量化）:
  每个向量存储: packed_values + group_mins + group_scales + radius
  4-bit, D=128, G=4: 64 + 8 + 8 + 2 = 82 bytes → 压缩比 3.12x
  
  注意力时每 token 读取: 82 bytes（含 group params）
  group params 占总读取量的 20%

TurboQuant（Lloyd-Max codebook 量化）:
  每个向量存储: packed_indices + radius
  4-bit, D=128: 64 + 2 = 66 bytes → 压缩比 3.88x
  
  注意力时每 token 读取: 66 bytes
  codebook 只有 16 个 float（64 bytes），完全在 L1 cache
  零 per-token overhead

差距: 3.12x vs 3.88x（压缩比），82 vs 66 bytes/token（带宽）
```

## 创新方案: Lloyd-Max Codebook + Warp Shuffle 融合 Kernel

### 核心思想

1. 随机旋转后，方向向量的每个坐标服从 Beta((d-1)/2, (d-1)/2) 分布
2. 这个分布是已知的，可以预计算最优 Lloyd-Max codebook
3. 量化时只需查表（不需要 per-group min/max）
4. 注意力 kernel 中，centroid table 加载到 shared memory，所有 token 共享

### 预期效果

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 4-bit 压缩比 | 3.12x | 3.88x | +24% |
| 2-bit 压缩比 | 5.12x | 10.4x | +103% |
| 每 token 读取 | 82 bytes | 66 bytes | -20% |
| 注意力速度 | 0.63x (seq=16K) | ~0.8x | +27% |

### 实现步骤

1. 实现 Lloyd-Max codebook 生成（Beta 分布 + 迭代优化）
2. 修改压缩 kernel：旋转 → 归一化 → codebook 查表量化
3. 修改注意力 kernel：centroid table 在 shared memory，indices 查表解压
4. 更新 benchmark 验证
