---
title: Mem0工作流踩坑记录
date: 2026-04-01
categories: 学习笔记
tags:
  - 大模型
  - Memory
  - workflow
---



# 引言

最近在折腾大模型的“记忆”能力。简单调研了一圈，发现 Mem0 的设计思路相当优雅：不仅支持自动提取，还能对记忆进行去重和融合，已经不只是“向量存储”那么简单。

不过，它的官方文档有一个比较明显的问题：

- Quickstart 主要围绕调用 OpenAI 等云端 API
- 关键细节较少
- 核心机制讲得比较零散

如果你和我一样，执着于**纯本地化部署**（模型、数据、记忆全部不出本地），那在跑通 Mem0 工作流的过程中，大概率会踩到几个不太明显的坑。

本文记录一个完整的、本地化的实践方案：

> 使用 **vLLM（Qwen3-14B 推理） + Ollama（Embedding） + Chroma（向量库）** 搭建 Mem0 本地记忆系统

# 架构选型与模型部署

我自己的目标很明确：实验一个简单的流程，给本地大模型加一个“长期记忆模块”，并且确保**所有数据完全本地存储**。

## 1. 对话模型：vLLM + Qwen3-14B

我这里选择用 vLLM 来部署对话模型，原因很简单：

- 笔者本身比较习惯使用vLLM来部署模型服务

- vLLM本身机制的优势

  - PagedAttention 对长上下文非常友好

  - 高并发能力强

  - 14B 模型也能跑出不错的速度

实际环境是一张 H20，跑 Qwen3-14B 基本可以做到“响应流畅”。

------

## 2. 向量模型：Ollama + nomic-embed-text

Embedding 这块，Mem0 支持很多方式，但我最终选了 Ollama：

- 部署简单（基本零配置），相比之下，用 vLLM 跑 embedding 有点“杀鸡用牛刀”。
- 自带 API
- **最关键：会自动释放显存**（这一点真的很香）

模型使用：`nomic-embed-text`，可以直接通过下面命令部署嵌入模型（确保Ollama已安装）：

```bash
ollama pull nomic-embed-text
```

可以使用 `curl` 简单测试一下 Ollama 的 API 是否能正常输出向量：

```
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Hello, world!"
}'
```

------

## 3. 向量存储：Chroma（SQLite）

为了轻量和方便，直接用 Mem0 自带的 Chroma：

- 本地文件存储（SQLite）
- 开箱即用
- 适合单机场景

------

## 4. Mem0 配置的关键点

Mem0 实际上依赖三块：

1. **LLM（用于“记忆提取与整理”）**
2. **Embedding（用于向量化）**
3. **Vector Store（用于存储）**

⚠️ 如果你**不显式配置 LLM**，Mem0 默认会调用 OpenAI —— 这就是后面第一个坑。

# Mem0 配置的“暗坑”

下面是整个过程中踩坑的几个点

## 暗坑 1：被忽略的“记忆提取器”

部署采用的全是本地模型，但一执行 `memory.add()`，直接报错缺少 OpenAI API Key。

这是因为 Mem0 并不是简单存文本。它在写入时会：

- 调用 LLM 分析对话
- 提取结构化信息
- 自动去重 & 合并历史记忆

如果你没配置 LLM，它就默认用 `gpt-4o`。必须显式指定本地 LLM，例如 vLLM：

```
"llm": {
    "provider": "openai",
    "config": {
        "model": MODEL_NAME,
        "openai_base_url": LOCAL_LLM_URL,
        "api_key": "none"
    }
}
```

## 暗坑 2：Chroma 维度的降维打击

如果你像我一样，最开始用了 HuggingFace 的 `multi-qa-MiniLM` (384维)，后来中途换成了 Ollama 的 `nomic-embed-text` (768维)。程序瞬间原地爆炸，报 Dimension Mismatch。这是因为同一个 Chroma 数据库文件夹一旦初始化了维度，就锁死了。换 Embedder 必须**删掉旧的数据库文件夹**重来。



# 核心代码实现 

下面是整理后的完整工作流代码（已踩坑验证版）。

在正式贴代码之前，先补充一个我在实践中加的小优化：**前置过滤函数**。它本质上是一个比较粗粒度的“熔断器”，用于拦截明显没有信息量的输入，避免无意义地触发记忆提取流程。

需要说明的是，这种做法更多是一个**工程层面的折中方案**，而不是一个“优雅”的最终解法。理想情况下，“什么该记、什么不该记”应该由模型来判断，而 Mem0 也确实在内部做了这件事。

具体来说，当调用 `memory.add()` 时，Mem0 并不是简单地存储文本，而是会触发一次 LLM 推理，用于：

1. 从对话中提取“用户事实”
2. 判断这些信息是否值得长期存储
3. 对记忆进行结构化表达（并可能与历史记忆融合）

它本身具备一定的筛选能力，例如：

- 通常不会把“哈哈哈”这类无意义输入写入记忆
- 更倾向提取偏长期的信息（如偏好、习惯、身份等）

但这里有两个现实问题：

- **这种判断本身是有成本的**（每次都会触发一次 LLM 推理）
- **它并不是一个稳定的分类器**（存在一定随机性，也依赖 prompt 质量）

因此，在本地部署场景下，我还是额外加了一层轻量的前置过滤，用来避免明显的算力浪费。至于更精细的“记忆触发策略”，后面可以单独展开，这里先以“跑通流程”为主。

下面是整个小实验的核心代码实现：

```python
from mem0 import Memory
from openai import OpenAI

# ========= 1. 本地大模型配置 (vLLM) =========
LOCAL_LLM_URL = "http://localhost:15700/v1"
MODEL_NAME = "qwen3-14b" 

client = OpenAI(base_url=LOCAL_LLM_URL, api_key="none")

# ========= 2. Mem0 核心配置 (vLLM + Ollama + Chroma) =========
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": MODEL_NAME,
            "openai_base_url": LOCAL_LLM_URL,
            "api_key": "none",
            "max_tokens": 1500,
            "temperature": 0.1 # 记忆提取务必用低温度，避免模型胡言乱语
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    # 持久化存储走 Chroma (SQLite)
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_local_hybrid",
            "path": "./mem0_db" 
        }
    },
    "version": "v1.1" 
}

memory = Memory.from_config(config)

# ========= 3. 极简过滤 =========
def should_store(text: str) -> bool:
    """拦截无效闲聊，避免无谓的记忆提取消耗"""
    keywords = ["喜欢", "是", "叫", "爱", "讨厌", "习惯", "想", "觉得"]
    return any(k in text for k in keywords)

# ========= 4. Workflow 对话函数 =========
def chat(user_input: str, user_id="default_user"):
    # 1. 检索上下文记忆
    search_res = memory.search(user_input, user_id=user_id, limit=5)
    if search_res and "results" in search_res and len(search_res["results"]) > 0:
        memory_text = "\n".join([f"- {m['memory']}" for m in search_res["results"]])
    else:
        memory_text = "None"

    # 2. 组装 Prompt
    prompt = f"""
You are a helpful AI assistant.
Here is relevant memory about the user:
{memory_text}

Use the memory to personalize your response if helpful.

User: {user_input}
Assistant:
"""

    # 3. vLLM 推理生成
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()

    # 4. 异步/条件写入记忆
    if should_store(user_input):
        messages_to_store = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer}
        ]
        memory.add(messages_to_store, user_id=user_id)

    return answer

# ========= 5. CLI循环 =========
if __name__ == "__main__":
    print("🤖 Memory Chat (type 'exit' to quit)\n")
    user_id = "user_1"

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        reply = chat(user_input, user_id=user_id)
        print(f"AI: {reply}\n")
```

首先先和模型自然对话，让模型能结合Memory机制记住我们说了什么。

![image-20260401140025241](https://files.seeusercontent.com/2026/04/01/9dUm/image-20260401140025241.png)

再次重新和模型对话：

![image-20260401142412723](https://files.seeusercontent.com/2026/04/01/0nEd/image-20260401142412723.png)



# 如何偷窥你的 Chroma 数据库？

代码跑通了，但是由于我们用了 Chroma 存在本地的 `./mem0_db` 文件夹里，它是一堆 SQLite 和二进制文件，因此看不到底层数据

如果你想亲眼看看 Mem0 到底给你的对话提取了什么特征，最简单的方式是再写一个脚本，直接用 Mem0 的原生 API 读取全部数据：

```python
from mem0 import Memory
# 复用上面的 config 字典...
memory = Memory.from_config(config)

all_memories = memory.get_all(user_id="hacker_01")
if all_memories and "results" in all_memories:
    for m in all_memories["results"]:
        print(f"🧠 [记忆]: {m['memory']} | 🕒 [时间]: {m['created_at']}")
```

下面是记忆的示例：

![image-20260401140055317](https://files.seeusercontent.com/2026/04/01/Uu5k/image-20260401140055317.png)

# 总结

关于“何时存、何时取”，一直是 Memory 系统中的核心问题之一。本次实验更多是一次“跑通链路”的实践记录，整体设计仍然偏向简单和直观，距离一个完善的记忆系统还有不小的提升空间。

后续会围绕这一方向做更深入的探索，例如：如何在工程上提升检索精度、进一步拆解 Mem0 的内部机制，以及不同 Memory 框架之间的对比与取舍等。

