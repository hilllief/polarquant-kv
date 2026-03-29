# TurboQuant 论文深度分析 + 超越方案

## 论文核心算法

TurboQuant（ICLR 2026, Google Research）的两阶段方法：

```
Stage 1: PolarQuant（MSE 最优量化）
  1. 随机正交旋转 → 每个坐标服从 Beta((d-1)/2, (d-1)/2) 分布
  2. 分布已知 → 预计算 Lloyd-Max 最优 codebook（零 per-block overhead）
  3. 查表量化 → 每个坐标只需 1 个 uint8 index

Stage 2: QJL（内积修正）
  1. 对量化残差做 1-bit JL 投影
  2. 存储符号位（每 token 额外 m bits）
  3. 注意力计算时用符号位修正 score → 无偏内积估计
```

论文声称：3.5-bit 零精度损失，2.5-bit 极小损失，H100 上 8x 加速。

---

## 论文的 5 个局限性

### 1. 只压缩 Key，不压缩 Value

论文和所有已知实现都只压缩 K，V 保持 FP16。理由是"softmax@V 是 compute-bound"。

但这只在短序列时成立。长序列（≥8K）时 V 的读取量 = seq × D × 2 bytes，也是带宽瓶颈。只压缩 K 意味着显存节省只有 ~50%（K 压缩了但 V 没有）。

### 2. 8x 加速仅限 H100 的 int4 tensor core

论文的 8x 加速数据来自 H100 的 int4 tensor core 指令。消费级 GPU（RTX 4090/5060 Ti）没有 int4 tensor core，实际加速远低于 8x。dejan.ai 的 RTX 4090 实现只达到 1.2x 加速。

### 3. QJL 修正的实现陷阱

论文描述了 TurboQuant_prod（Lloyd-Max + QJL），但实践中发现：
- QJL 修正不能加回重建向量（会注入噪声，cos 降到 0.69）
- QJL 只在自定义注意力 kernel 中有效
- 论文没有充分讨论这个陷阱

### 4. 没有 Value 的融合 kernel

所有实现只融合了 Q@K^T 的计算。softmax@V 仍然用标准 FP16 matmul。如果 V 也压缩，需要第二个融合 kernel，这是未解决的工程问题。

### 5. codebook 查表的 gather 效率

Lloyd-Max codebook 查表是 gather 操作（随机内存访问）。GPU 上 gather 效率低于连续内存访问。论文没有讨论 L1/L2 cache 利用策略。

---

## 超越论文的 5 个创新方向

### 创新 1: K+V 双压缩 + 双融合 Kernel（最高优先级）

```
论文: 只压缩 K → 显存节省 ~50%
我们: K+V 都压缩 → 显存节省 ~75%

实现方案:
  - K 融合 kernel: 从 packed indices 查 codebook → score
  - V 融合 kernel: 从 packed indices 查 codebook → weighted sum
  - 两个 kernel 串联，中间只传 softmax weights（seq 个 float）
  - V 不需要完整解压到 FP16 tensor

预期效果:
  - 显存: 从 50% 节省 → 75% 节省
  - 带宽: 从 2x 节省（只 K）→ 4x 节省（K+V）
  - 速度: 长序列可能达到 2x+ 加速
```

### 创新 2: 自适应 Per-Layer Bit-Width

```
论文: 所有层用相同 bit-width（如全部 4-bit）
我们: 根据层的重要性动态分配 bit-width

方法:
  1. 用少量 calibration 数据测量每层的 attention entropy
  2. 高 entropy 层（信息分散）→ 用 4-bit（需要高精度）
  3. 低 entropy 层（信息集中在少数 token）→ 用 2-bit
  4. 平均 bit-width 可以降到 ~3-bit，但精度接近全 4-bit

预期效果:
  - 相同精度下压缩比提升 30%
  - 或相同压缩比下精度提升
```

### 创新 3: Token 重要性感知量化

```
论文: 所有 token 用相同精度
我们: 重要 token 高精度，不重要 token 低精度

方法:
  1. 用 attention score 的历史统计判断 token 重要性
  2. "sink tokens"（开头几个 token）和最近的 token → 4-bit
  3. 中间的 token → 2-bit
  4. 结合 ThinKV 的思路：渐进式精度衰减

预期效果:
  - 平均 bit-width 降到 ~2.5-bit
  - 精度接近全 4-bit（因为重要 token 保持高精度）
```

### 创新 4: 跨层差分编码

```
论文: 每层独立压缩
我们: 利用相邻层 KV 的相似性

方法:
  1. 第 0 层存完整的量化 KV
  2. 第 1 层存与第 0 层的差值（差值更小，量化误差更低）
  3. 差值可以用更少的 bit 表示

预期效果:
  - 额外 20-30% 的压缩比提升
  - 需要验证跨层相似性假设
```

### 创新 5: Blackwell 架构特化

```
论文: 基于 H100 (Hopper) 优化
我们: 基于 RTX 5060 Ti (Blackwell, sm_120) 优化

Blackwell 的新特性:
  - 更大的 L2 cache（48MB vs H100 的 50MB，但消费级更便宜）
  - 新的 warp 调度策略
  - 可能支持 FP4 tensor core

优化方向:
  - 利用大 L2 cache 缓存 codebook 和 group params
  - 针对 sm_120 的 warp 调度优化 kernel
```

---

## 实施优先级

| 优先级 | 创新 | 预期提升 | 工程量 | 风险 |
|--------|------|---------|--------|------|
| P0 | K+V 双融合 kernel | 速度 2x, 显存 75% | 2 天 | 低 |
| P1 | 自适应 bit-width | 压缩比 +30% | 3 天 | 中 |
| P2 | Token 重要性感知 | 压缩比 +50% | 5 天 | 中 |
| P3 | 跨层差分编码 | 压缩比 +20% | 3 天 | 高 |
| P4 | Blackwell 特化 | 速度 +20% | 5 天 | 高 |

---

## 我们的当前优势

1. **K+V 双压缩**：论文只压缩 K，我们已经压缩了 K 和 V
2. **消费级 GPU 优化**：论文只在 H100 上测试，我们在 RTX 5060 Ti 上实现
3. **全 C++ 融合 kernel**：绕过 PyTorch 兼容性问题，零 Python overhead
4. **完整的 TDD 测试套件**：154 个测试，算法正确性有保障
5. **Lloyd-Max + warp shuffle**：结合了论文的算法和我们的 GPU 优化
