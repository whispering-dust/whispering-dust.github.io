---
title: MemFactory 框架探索笔记
date: 2026-04-03
categories: 学习笔记
tags:
  - 大模型
  - Memory
  - RL
---


最近一直在研究用RL去训练LLM的Memory能力，也计划复现一些文章看看效果，恰好找到一个叫 MemFactory 的框架，它是一个用强化学习（GRPO）训练 LLM 记忆处理能力的框架，并且集成了几个代表性的论文的训练，比如Memory-R1和MemAgent。于是花了点时间把代码认真读了一遍，一方面熟悉框架，一方面也从代码层面熟悉代表性的RL4Mem的研究。此外也尝试跑通了训练，这里记录一下整个探索过程。

---

## 1. 框架概览

> [MemFactory官方仓库](https://github.com/Valsure/MemFactory)

![logo](https://files.seeusercontent.com/2026/04/03/woG0/overall.png)

MemFactory 要解决的问题可以用一句话概括：**用强化学习训练 LLM 处理长期记忆的能力**。

人和 LLM 交互的过程中，有大量信息值得被记住——用户的偏好、过去发生的事情、重要的约定。但普通的 LLM 没有这种持久记忆机制，每次对话都从零开始。MemFactory 的思路是训练模型去主动做这件事：从对话中抽取记忆、更新记忆库、在需要回答时检索相关记忆。为了训练这个能力，框架实现了一套基于 GRPO 的强化学习训练流程。整体分三层：

```
Trainer (MemGRPOTrainer)
  |-- Agent（编排各模块，产出训练样本 Samples）
  |     |-- Module: Extractor（从对话中抽取结构化记忆）
  |     |-- Module: Updater（决定如何合并新旧记忆）
  |     |-- Module: Retriever（检索相关记忆用于回答）
  |-- Env（PyTorch Dataset + 奖励计算）
```

三层之间的分工非常清晰：Env 负责提供数据和定义什么算"做对了"，Agent 负责让模型生成轨迹，Trainer 负责用这些轨迹更新模型参数。

框架目前有三种 Agent 实现，对应三种不同的研究方向：

- **MemoryAgent（memagent）**：最简单的一种，模型循环读取长文本的每个 chunk，逐步更新记忆，最终基于记忆回答问题。这是论文的主要复现路径。
- **Memory-R1（memory_r1_agent）**：把抽取和更新两个步骤都做成可训练模块，联合训练。
- **RMM（memory_rmm_agent）**：抽取和更新用 API 推理，只训练检索器（reranker）。

---

## 2. 框架实现原理深度解析

### 2.1 记忆的来源与表示

> 原文整理：
>
> - [MemAgent](https://arxiv.org/abs/2507.02259)
> - [Memory-R1](https://arxiv.org/abs/2508.19828)
> - [RMM](https://arxiv.org/abs/2503.08026) 

在 MemoryAgent 这条路线里，记忆的形态非常朴素：**就是一段文本**，存在 Python 变量里，没有数据库，没有向量索引。

训练数据的格式是 `{context, question, answers}`，context 是一篇长文档。训练目标是让模型学会从长文档里提炼出能帮助回答 question 的关键信息，并把这些信息"压缩"进一段记忆文本里。

在 Memory-R1 和 RMM 路线里，记忆变成了结构化的 `MemoryItem` 对象：

```python
@dataclass
class MemoryItem:
    id: str
    key: str          # 记忆标题
    value: str        # 记忆内容
    memory_type: str  # LongTermMemory / UserMemory
    tags: List[str]
    ...
```

这些记忆条目存储在 Neo4j（结构化数据）和 Milvus（向量索引）里，支持语义检索。

不过这套存储层在框架内有完整的 mock 实现，训练时默认走内存存储，不需要真实的数据库。这是一个很神奇的设计点，对于**MemoryAgent** 来说，这个问题根本不存在。它的记忆只是一个字符串变量，在 rollout 的循环里直接拼进 prompt，整个存储层完全没有被调用过，mock 对它零影响。

真正有问题的是 **Memory-R1** 和 **RMM**。这两条路线的训练流程是：模型输出 JSON 格式的操作指令 → 写入 mock 存储 → 从 mock 存储检索（随机向量，语义无关）→ 拼进 prompt 回答问题 → LLM Judge 打分。检索结果是随机的，模型拿到的上下文是乱的，奖励信号自然无法反映"这条记忆操作是否真的有用"。但其实代码里注释说明了这一点：

```python
# NOTE: Currently, this reward primarily reflects JSON format compliance
# rather than downstream reasoning accuracy.
```

所以 mock 的真实意图变得清晰了：它是一个**工程上的占位符**，让框架在**没有任何外部依赖的情况下能跑通完整流程**，验证代码逻辑，同时也把接口留好，**等待使用者用真实服务替换**。Memory-R1 / RMM 路线如果要认真训练，需要自己部署 Embedding 服务和向量数据库，让检索有真实的语义相关性，奖励信号才能真正指导模型学会有意义的记忆管理策略。这部分框架提供了完整的接口，但没有帮你填上。

### 2.2 三种 Agent 的 Rollout 机制

三种 Agent 的 rollout 流程差别很大，对应着完全不同的训练目标。

#### 2.2.1 MemoryAgent：循环读取，压缩记忆

MemoryAgent 的核心可以理解为模型在"做笔记"：把一篇长文档切成若干 chunk，每读一块就更新一次记忆文本，最终基于积累的记忆回答问题。

```
长文档按 chunk_size 切块（默认 2500 tokens，最多 6 块）

  memory_0 = "No previous memory"

  chunk_1 → model(question + memory_0 + chunk_1) → memory_1
  chunk_2 → model(question + memory_1 + chunk_2) → memory_2
  ...
  chunk_N → model(question + memory_{N-1} + chunk_N) → memory_N

  最终：model(question + memory_N) → \boxed{answer}
```

记忆在这里就是一个字符串变量，每步直接拼进 prompt，没有数据库，没有检索。对同一个样本，模型并行生成 16 条独立轨迹（`num_generations=16`），每条轨迹各自走完完整的 chunk 读取过程，最终给出 16 个不同的答案，供 GRPO 做组内比较。

#### 2.2.2 Memory-R1：两阶段生成，联合训练

Memory-R1 把记忆处理拆成"抽取"和"更新"两个显式步骤，两步都用本地模型生成，都参与训练。

```
输入：fact（一段对话历史）+ context_memory（已有记忆库）

Step 1 - Extractor 生成抽取结果：
  model(fact) → JSON { "memory_list": [...] }

Step 2 - Updater 生成更新指令：
  model(context_memory + extraction_output) → JSON { "operations": [...] }
  operations 中每条记忆都有对应操作：ADD / DEL / UPDATE / NONE
```

两步各自产出一批 `(prompt, response)` 对，打包成两个独立的 Samples 返回给 Trainer：

```python
return {
    "extraction": ext_samples,
    "update": upd_samples
}
```

Trainer 根据 `train_extraction` 和 `train_update` 两个开关，可以选择只训练其中一步，或两步同时训练。

#### 2.2.3 RMM：固定前两步，只训练检索器

RMM 的设计思路更聚焦：抽取和更新用 LLM API 推理（不训练），只训练检索器（Reranker）。

```
Step 1 - Extractor.inference()：调 API，固定不训练
Step 2 - Updater.inference()：调 API，固定不训练
Step 3 - Retriever.rollout()：本地模型生成，这才是训练目标
```

Retriever 的任务是：拿到一批候选记忆条目，从中选出最值得用于回答问题的几条，输出它们的 ID 列表，格式如 `[1, 3, 5]`。选得好，后续 QA 就能答对，奖励就高。

这个设计的出发点是：抽取和更新是相对明确的 NLP 任务，可以直接用强模型 API 完成；而检索重排序依赖对"什么样的记忆真正有用"的深层理解，更适合用 RL 来训练。

### 2.3 三种 Agent 的奖励设计

三种 Agent 分别对应不同的 Env，奖励机制各不相同。

#### 2.3.1 MemoryAgent → LongContextMemoryEnv：纯二值奖励

奖励计算分两步：

第一步，本地格式检查（零 API 成本）：

```python
boxed_content = extract_boxed_content(response)
if boxed_content is None:
    return 0.0  # 没有 \boxed{}，直接 0 分，不调 Judge
```

第二步，调 LLM Judge 做语义判断：

```
Question: {question}
Standard Answer: {answer}
Predicted Answer: {prediction}

Is the predicted answer consistent with the standard answer? Please output only "True" or "False".
```

最终奖励是纯二值：答对 1.0，答错 0.0。Judge 的任务只是判断语义等价，不需要有领域知识，3B 左右的小模型就够胜任。

#### 2.3.2 Memory-R1 → MemoryBankEnv：格式奖励为主

抽取和更新各自独立打分，奖励标准相同：输出了合法 JSON 且包含对应的必要键，给 0.5 分，否则 0 分。

```python
# 抽取奖励
ext_reward = 0.5 if (ext_json != {} and "memory_list" in ext_json) else 0.0

# 更新奖励
upd_reward = 0.5 if (upd_json != {} and "operations" in upd_json) else 0.0
```

代码里其实写好了更完整的准确率奖励逻辑——执行操作指令、检索记忆、调 LLM 回答问题、Judge 打分——但最终这部分被注释掉了：

```python
# NOTE: Currently, this reward primarily reflects JSON format compliance
# rather than downstream reasoning accuracy.
# if accuracy_reward > 0:
#     final_ext += accuracy_reward
#     final_upd += accuracy_reward
```

所以当前这条路线训练的主要是"输出格式正确的 JSON"，而不是"操作真的有用"。这和前面提到的 mock 问题是同一个根源：没有真实的语义检索，准确率奖励信号太噪，干脆先关掉。

#### 2.3.3 RMM → RerankBankEnv：三层叠加奖励

RMM 的奖励是三层叠加，设计最精细：

```
格式奖励（0.5）：输出了合法的 ID 列表，且选择数量在 1~8 之间

准确率奖励（1.0）：基于选出的记忆，LLM 能正确回答问题

引用奖励（最多 1.0）：回答里引用了选中记忆的 ID
                       每个正确引用 +0.125，最多 8 个（= 1.0）
```

最高总分 2.5 分。引用奖励的设计比较有意思——它鼓励模型不只选对记忆，还要在回答时明确指出用了哪条记忆，类似"有据可查"的要求，防止模型胡乱检索但凑巧答对。

### 2.4 GRPO 训练流程

有了 16 条轨迹和对应的奖励之后，GRPO 的核心计算是组内归一化：

```python
scores_tensor = torch.tensor(scores)          # 16 个 0.0 或 1.0
mean_score = scores_tensor.mean()
std_score = scores_tensor.std()

# 如果 16 条全对或全错，std=0，跳过这个 batch
if std_score.item() < 1e-6:
    continue

advantages = (scores_tensor - mean_score) / (std_score + 1e-8)
```

这是 GRPO 区别于 PPO 的关键：不需要训练一个 value function，而是用**同一 batch 内的相对表现**来估计 advantage。答对率高的轨迹 advantage > 0，答错的 < 0。

整个框架是纯 PyTorch 实现，没有用 DeepSpeed 或 TRL。模型在每个 batch 里经历三次前向传播：

1. `model.generate()` —— 采样 16 条轨迹（no_grad）
2. `model.forward()` —— 计算 old log probs 作为重要性采样基准（no_grad）
3. `model.forward()` —— 计算 new log probs，loss 从这里反传（**有梯度**）

Loss 的计算是 PPO clip + KL 惩罚的组合：

```python
ratio = exp(new_log_probs - old_log_probs)         # 重要性采样比
clipped_ratio = clamp(ratio, 1 - eps, 1 + eps)     # clip 防止更新过猛
per_token_loss = -min(ratio * advantage, clipped_ratio * advantage)
per_token_loss += beta * KL(ref_model || current_model)  # KL 惩罚
```

其中 `ref_model` 是训练开始时 deepcopy 的初始模型，训练过程中永远不更新，作为 KL 惩罚的锚点，防止策略跑偏太远。

只有 `action_mask=True` 的 token（即模型生成的 response 部分）才参与 loss 计算，prompt 部分不参与。

---

## 3. 跑通训练

对于 MemoryAgent 这条路线（`memagent` + `longcontext`），真正需要的东西只有三样：

1. **本地模型权重**：Qwen3-1.7B 或 4B，下载到本地
2. **训练数据**：HotpotQA 转换版，官方提供了 [下载路径](https://huggingface.co/datasets/nworats/MemFactory)
3. **LLM Judge 服务**：任意 OpenAI 兼容的 API，本地 ollama 或远程服务都行，我在这里使用vllm部署了一个qwen3-4B的小模型用于计算reward（reward计算不算复杂，同时我不追求完全复现原本的效果）

`.env` 里关于 Embedding、Neo4j、Milvus 的配置，由于我只复现MemAgent ，因此不需要填——记忆在MemAgent只是一段文本变量，根本不走存储层。

训练启动命令：

```bash
bash examples/RunMemoryAgent1.7B.sh
```

关键超参数都在 shell 脚本里，主要的几个：

```bash
NUM_GENS=16          # 每个样本生成的轨迹数
MAX_PROMPT_LEN=6000  # prompt 最大长度
MAX_GEN_LEN=2500     # 每步最大生成长度
chunk_size=2500      # 每个 chunk 的 token 数
max_chunk_number=6   # 最多处理几个 chunk
```

我训练时采用的是单卡 H20，按默认配置（1.7B 模型，2000条数据，epoch=2），并另外用一张卡部署qwen3-4B作为judge model。但是由于环境的问题，训练的过程极其缓慢。于是我尝试分析了一下训练速度慢的原因，并重新整理了一些加速的常见概念，单独整理在了 [训练加速探索笔记](./0403-train_speedup.md) 里。

---

## 4. 总结

读完这个框架最大的感受是：**把复杂的问题拆得很干净**。

GRPO 训练 LLM 记忆能力这件事，听起来很复杂，但框架把它拆成了互相独立的几个部分：Env 只管数据和奖励定义，Agent 只管轨迹生成，Trainer 只管梯度更新，彼此之间通过 Samples 这个数据结构传递，耦合极低。

整个实现没有依赖任何重型训练框架，GRPO 的核心逻辑就是几十行 PyTorch，但逻辑却是完整的。对于想理解 RLHF/GRPO 原理的人，这个代码库是一个非常好的切入点——足够小，足够清晰，又是真实可以跑起来的实现。此外，想要自定义任何一个环节，只需要继承对应基类、注册、实现接口，完全不需要动其他地方，给使用者提供了足够的便利，真心挺不错的。

