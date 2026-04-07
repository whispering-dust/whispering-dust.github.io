---
title: 训练加速探索：记一次DEBUG和加速原理的总结
date: 2026-04-03
categories: 学习笔记
tags:
  - 大模型
  - 训练加速
  - Flash Attention
  - RL
  - DEBUG
---

在跑 MemFactory 训练的过程中，发现 rollout 阶段非常慢，追查下来是 attention 实现的问题。这里记录一下排查过程和最终解决方案。

---

## 1. 问题发现

训练启动后日志里出现了这行：

```
Flash Attention 2 not found, using default attention
```

这意味着模型加载时 Flash Attention 2 没有生效，attention 计算走的是 transformers 默认的 eager 实现，复杂度是 O(n²)。

MemoryAgent 的 rollout 序列长度相当长：

```
prompt = 模板 + 问题 + 旧记忆 + chunk
       ≈ 最多 6000 tokens（max_prompt_length 配置）
```

在 6000 tokens 的序列长度下，eager attention 的计算开销非常可观，而 rollout 时 batch size 又是 16（num_generations），每个 batch 还要做 5~7 次 generate。这是 rollout 慢的直接原因。

---

## 2. 尝试安装 Flash Attention 2

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

安装失败，错误信息：

```
Guessing wheel URL: https://github.com/.../flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-...whl
Precompiled wheel not found. Building from source...
...
RuntimeError: The detected CUDA version (13.1) mismatches the version that was
used to compile PyTorch (12.8).
```

### 2.1 根本原因

环境里存在两套 CUDA：

| | 版本 |
|---|---|
| 系统 CUDA（`nvcc` 指向的） | 13.1 |
| PyTorch 编译时使用的 CUDA | 12.8（torch 版本为 2.9.1+cu128） |

flash-attn 在编译 C++ extension 时，`torch/utils/cpp_extension.py` 内部会做版本一致性检查，发现两者不匹配，直接抛出 RuntimeError 终止编译。这是 PyTorch 的安全机制，不是 flash-attn 的问题。

预编译 wheel 也不存在（HTTP 404）：torch 2.9.1+cu128 是比较新的组合，flash-attn 官方尚未发布对应的预编译包，只能从源码编译，而源码编译又因为上面的问题失败了。

### 2.2 理论上的修复路径

如果 conda 环境里附带了和 torch 编译一致的 CUDA toolkit，可以临时把它放到 PATH 最前面绕过系统 nvcc：

```bash
# 找 conda 环境里的 nvcc
find /path/to/conda/envs/MemFactory -name "nvcc" 2>/dev/null

# 临时覆盖
export CUDA_HOME=/path/to/conda/envs/MemFactory
export PATH=$CUDA_HOME/bin:$PATH

MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

但这需要 conda 环境里确实安装了 cudatoolkit，且版本和 torch 一致，不一定能成功。

---

## 3. 替代方案：sdpa

PyTorch 2.0 之后内置了 `scaled_dot_product_attention`（sdpa），在 CUDA 上会自动选择最优 kernel，包括类 FlashAttention 的 memory-efficient 实现。不需要安装任何额外的包。

修改 `examples/train_mem_grpo.py`，在 Flash Attention 不可用时 fallback 到 sdpa 而不是 eager：

```python
# 修改前
try:
    import flash_attn
    model_kwargs["attn_implementation"] = "flash_attention_2"
    print("Using Flash Attention 2")
except ImportError:
    print("Flash Attention 2 not found, using default attention")
    # 没有设置 attn_implementation，走 eager 默认实现

# 修改后
try:
    import flash_attn
    model_kwargs["attn_implementation"] = "flash_attention_2"
    print("Using Flash Attention 2")
except ImportError:
    print("Flash Attention 2 not found, using sdpa attention")
    model_kwargs["attn_implementation"] = "sdpa"   # 显式指定 sdpa
```

### 3.1 效果

改用 sdpa 后，rollout 速度提升约 **1.5 倍**。

三种 attention 实现的对比：

| 实现 | 复杂度 | 需要安装 | 实测加速（相对 eager） |
|------|--------|---------|----------------------|
| eager（默认） | O(n²) 显存 | 不需要 | 1x（基准） |
| sdpa | O(n) 显存（memory-efficient kernel） | 不需要，PyTorch 内置 | ~1.5x |
| Flash Attention 2 | O(n) 显存，IO-aware 优化 | 需要编译安装 | ~2~4x |

sdpa 达不到 Flash Attention 2 的水平，主要原因是 FA2 在 IO 层面做了更激进的优化（减少 HBM 读写次数），而 sdpa 的 memory-efficient kernel 主要解决的是显存占用问题，吞吐量提升有限。

### 3.2 关于 Judge 速度的补充

除了 attention 本身，另一个影响 rollout 速度的因素是 LLM Judge 的调用效率。

代码里用了 `ThreadPoolExecutor(max_workers=16)` 并发发出 16 个请求，但 ollama 默认是单请求队列，实际是串行处理的。如果 Judge 模型用 CPU 推理，单次 2~3 秒，16 次串行就是 30~50 秒/batch，远超 GPU 计算时间。

改善方式：
- 使用 0.5B 的极小模型做 Judge（单次 < 0.5s，16次串行也只需 ~8s）
- 设置 `OLLAMA_NUM_PARALLEL=16` 启用 ollama 并发处理
- 换成真正支持并发的远程 API（如 DeepSeek、硅基流动等）

Judge 任务本质只是判断两段文本语义是否等价，不需要大模型，选最小够用的模型在这里收益最大。


---

## 4. 加速原理整理

通过这次排查，顺便把 LLM 推理和 RL 训练里常用的加速手段梳理了一遍，重点放在和这次遇到的问题直接相关的部分。

### 4.1 Attention 计算的瓶颈在哪里

标准的 Scaled Dot-Product Attention 公式是：

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

计算 `QKᵀ` 的结果是一个 `[seq_len, seq_len]` 的矩阵，这带来两个问题：

**显存问题**：这个矩阵的大小是序列长度的平方。seq_len=6000 时，单个 attention head 的中间矩阵就需要 6000×6000×2 bytes（fp16）≈ 72MB，乘以层数和 head 数，显存很快爆掉。

**速度问题**：GPU 计算本身很快，真正的瓶颈是**数据在显存（HBM）和 SRAM（片上缓存）之间的搬运**。标准实现需要把 `QKᵀ` 的结果写回 HBM，做完 softmax 再读回来和 V 相乘，这两次大矩阵的读写是主要开销，而不是实际的乘法运算。

这就是为什么 eager attention 在长序列下特别慢——计算量是 O(n²)，IO 也是 O(n²)，两者叠加。

### 4.2 sdpa 做了什么

PyTorch 的 `F.scaled_dot_product_attention` 在 CUDA 上会根据环境自动选择后端 kernel，主要有三个：

- **FlashAttention**：如果安装了 flash-attn 库
- **memory-efficient attention**（来自 xformers）：如果安装了 xformers
- **math**：纯 PyTorch 实现，即 eager 的等价版本

在什么都没装的情况下，CUDA 上的 sdpa 会走 **memory-efficient attention** 的内置实现。这个实现的核心思路是：把 `[seq_len, seq_len]` 的 attention 矩阵分块计算，每次只在片上 SRAM 里处理一小块，不把完整的中间矩阵写回 HBM。这样显存占用从 O(n²) 降到 O(n)，同时减少了 HBM 读写次数。

但它和 Flash Attention 2 的差距在于：FA2 对分块策略和 IO 调度做了更激进的优化，能更充分地利用 GPU 的异步计算能力，实际吞吐量更高。sdpa 的 memory-efficient kernel 主要解决的是显存问题，速度提升是副产品，所以实测只有 1.5 倍左右，而 FA2 通常能到 2~4 倍。

用一句话概括两者的区别：**sdpa 让你用更少显存跑起来，FA2 让你跑得更快**。

### 4.3 gradient checkpointing：用时间换显存

这个框架在 Trainer 初始化时默认开启了：

```python
if self.args.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

正常的反向传播需要在前向时把每一层的激活值全部保存下来，供梯度计算使用。对于 1.7B 的模型，这些激活值会占用大量显存，尤其是序列很长（6000+ tokens）的时候。

gradient checkpointing 的做法是：前向时**不保存**中间激活值，只保存关键的 checkpoint 节点（通常是每个 transformer block 的输入）。反向传播需要某层的激活时，从最近的 checkpoint 重新做一次前向计算。

代价是每个 batch 多了大约一次额外的前向计算，训练时间增加约 30%。但显存节省非常显著，对于长序列训练几乎是必开项。这次 rollout 序列最长 8500 tokens（6000 prompt + 2500 response），不开这个很可能直接 OOM。

### 4.4 bfloat16：训练时的精度选择

框架加载模型时自动选择精度：

```python
"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

H20 支持 bfloat16，会走 bfloat16 路径。bfloat16 和 float16 都是 16 位，但数值范围不同：

- float16：指数 5 位，尾数 10 位，数值范围小，训练时容易溢出，需要额外的 loss scaling
- bfloat16：指数 8 位（和 float32 相同），尾数 7 位，数值范围和 float32 一致，训练稳定，不需要 loss scaling

代码里也体现了这一点：

```python
# 只有 float16 才需要 GradScaler，bfloat16 不需要
self.scaler = torch.amp.GradScaler() if (device == 'cuda' and model.dtype != torch.bfloat16) else None
```

对于 RL 训练来说，bfloat16 比 float16 更合适——RL 的奖励信号本身就有噪声，精度影响不大，但数值稳定性更重要，bfloat16 的大数值范围能避免 advantage 或 log_prob 计算时出现 NaN。

### 4.5 RL 训练特有的速度问题：rollout 是瓶颈

普通的监督学习训练，瓶颈在反向传播。但 GRPO 这类 RL 训练，rollout 阶段的开销往往远大于训练阶段，原因是：

1. **rollout 是串行的自回归生成**：每个 token 都要做一次前向，没有办法像训练时那样并行处理整个序列
2. **生成量是训练量的数倍**：num_generations=16，每个样本生成 16 条轨迹，还要做 5~7 次 generate（每个 chunk 一次）
3. **外部依赖引入了等待**：Judge API 的调用是同步的，GPU 在等 CPU/网络

这也是工业界用 vLLM 做 rollout 的原因——vLLM 的 PagedAttention 和 continuous batching 能显著提升生成吞吐量，是 transformers 原生 `model.generate()` 的数倍。MemFactory 的评估阶段（`evaluate_worker.py`）就用了 vLLM，训练阶段用的是原生生成，这是一个可以优化的点。

