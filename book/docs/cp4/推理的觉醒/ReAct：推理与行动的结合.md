---
sidebar_position: 1
---


## 模型被困在了"想象"的牢笼里

2022年的夏天，当Chain-of-Thought论文开始在学术界掀起波澜时，一个新的问题也随之浮现：推理很重要，但推理的目的是什么？答案应该是行动——或者说，推理应该指导我们如何与外部世界互动。然而，当时的大多数研究都将CoT限制在纯粹的思维层面。模型生成一系列中间推理步骤，然后给出答案，整个过程完全封闭在模型的内部世界中。

这个问题在某些任务上表现得尤为突出。比如，当你问一个模型"2023年的美国总统是谁"时，CoT可能会帮助模型更逻辑清晰地推理，但如果其训练数据中对2023年的信息不完整或矛盾，推理本身无法解决这个核心问题。模型需要的不仅仅是更好的思考方式，而是突破模型参数的局限，连接到实时的、动态的外部信息源。

2022年的下半年，来自谷歌大脑的Shunyu Yao、Denny Zhou等研究者发表了一篇题为《ReAct: Synergizing Reasoning and Acting in Language Models》的论文。这篇论文提出了一个看似简单但深刻的想法：何不让模型在生成推理步骤的同时，也生成对应的行动指令？让模型告诉我们它想要采取什么行动，然后我们将行动的结果（Observation）反馈给模型，让它继续推理。这样就形成了一个闭环：Thought（想法）→ Action（行动）→ Observation（观察）→ Thought（新想法）→ ...

这个想法打破了大语言模型的"想象力监狱"。模型不再被限制在自己的参数知识范围内，而是可以通过行动来改变自己面对的世界状态。

## 三部曲：思想、行动、观察

ReAct框架的核心循环可以这样描述。首先，模型生成一个Thought，这个思想阐述了当前的情况分析和下一步的计划。接着，基于这个想法，模型生成一个Action，这个行动可以是调用某个工具、查询某个数据库、或者执行某个计算。然后，外部系统执行这个行动，并将Observation（观察结果）返回给模型。模型看到这个观察结果后，会更新自己的理解，生成新的Thought，进而生成新的Action。这个过程循环进行，直到模型认为已经获得了足够的信息来生成最终答案。

让我们用一个具体的例子来理解这个过程。假设我们问模型："在LeCun论文中关于自监督学习的观点，与Hinton关于深度学习的早期理论有什么关键区别？"这个问题需要模型查找并比较两位研究者的具体论文内容。使用ReAct框架，模型会生成类似这样的过程：

```
Thought: 我需要找到LeCun关于自监督学习的论文和Hinton的深度学习理论文章。
        让我先搜索LeCun的自监督学习工作。

Action: search("Yann LeCun self-supervised learning")

Observation: 找到了LeCun关于自监督学习的论文《A Path Towards Autonomous 
           Machine Intelligence》以及其他相关工作...

Thought: 好的，现在我有了LeCun的自监督学习观点。现在让我搜索Hinton的深度学习理论。

Action: search("Geoffrey Hinton deep learning theory foundations")

Observation: 找到了Hinton在《A Fast Learning Algorithm for Deep Belief Nets》
           以及其他深度学习基础工作的信息...

Thought: 现在我有了两位研究者的主要思想。让我比较它们的关键差异...
```

这个伪代码演示展示了ReAct如何创造了一个迭代的、可观察的推理过程。与Chain-of-Thought不同，CoT的所有推理都发生在模型内部，结果对用户来说是黑盒的。而ReAct通过明确的Action和Observation，使得推理过程变成了可交互的、可验证的。你可以在任何时刻打断这个循环，检查模型采取的行动是否合理，观察结果是否符合预期。

## 工具，而非知识

ReAct框架的实际威力体现在工具的集成上。在Yao等人的论文中，他们演示了将ReAct应用于各种需要外部工具的任务：维基百科搜索、计算器、代码执行环境等等。每个工具都被建模为一个"行动空间"中的可用操作。

以HotpotQA任务为例（一个需要多步骤推理和信息检索的问答数据集），模型可用的工具包括：

```python
# 伪代码：ReAct中的可用工具集合

class ToolSet:
    def search(self, query: str) -> str:
        """在知识库中搜索相关信息"""
        pass
    
    def lookup(self, entity: str) -> str:
        """查找特定实体的详细信息"""
        pass
    
    def calculate(self, expression: str) -> float:
        """执行数学计算"""
        pass
    
    def reason(self, statement: str) -> bool:
        """对逻辑陈述进行推理"""
        pass

# ReAct agent的执行伪代码
class ReActAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.history = []  # 记录完整的Thought-Action-Observation历史
    
    def step(self, prompt: str) -> str:
        # 将历史记录和当前任务作为上下文
        context = self._build_context()
        
        # 模型生成下一个Thought和Action
        response = self.model.generate(context + prompt)
        thought, action = self._parse_response(response)
        
        # 执行行动
        observation = self._execute_action(action)
        
        # 记录历史
        self.history.append({
            'thought': thought,
            'action': action,
            'observation': observation
        })
        
        return observation
    
    def _execute_action(self, action: str) -> str:
        """根据action类型调用相应工具"""
        if action.startswith("search("):
            query = action[7:-1]  # 提取查询字符串
            return self.tools.search(query)
        # ... 处理其他工具
```

这个设计的关键在于：模型不需要拥有所有知识，它只需要知道如何使用工具。一个接过搜索、计算器和代码执行的ReAct agent可以回答远超过其参数知识范围的问题。这似乎是一个微小的转变，但其影响是深远的——它意味着大语言模型的能力边界不再由其训练数据和参数大小硬性确定，而是由可用的工具集合所定义。

ReAct之所以有效，一个至关重要的因素是提示工程。研究者们发现，如何格式化Thought和Action的指令对最终性能有巨大影响。通常，最有效的格式包含以下元素：

首先是清晰的任务描述。告诉模型它可以使用哪些工具，以及每个工具应该如何调用。其次是具体的示例。少量的演示样本（few-shot）展示了Thought-Action-Observation的正确格式，这帮助模型理解期望的行为。最后是明确的停止条件。何时模型应该停止检索新信息并给出最终答案，这需要通过示例和指令明确说明。

论文中的一个有趣发现是，即使对于GPT-3这样已经相当强大的模型，提示格式的改变也能带来显著的性能提升。在HotpotQA上，标准的Few-shot In-Context Learning达到了约62%的准确率，而相同的模型使用ReAct框架则达到了71%。这个9个百分点的提升并非来自模型容量的增加或数据的增多，而纯粹来自于改变了模型与外部工具交互的方式。

更有趣的是，论文发现CoT + ReAct的组合往往优于单独的ReAct。这提示我们，让模型同时展示其内部推理过程（CoT的思维链）和外部行动（ReAct的工具调用），能够创造一种互补的效果。内部推理帮助模型规划它的行动序列，而外部行动则为内部推理提供了验证和新信息。

## 从学术突破到现实应用

ReAct框架的提出标志了一个重要的转折点。在此之前，大语言模型主要被研究者和应用开发者视为静态的预测系统——你向它提供输入，它生成输出，过程结束。而ReAct将其转化为一个动态的、交互式的系统。这为后来的Agent范式打下了基础。

实际上，ReAct被认为是现代AI Agent设计的先驱性工作。当OpenAI后来推出插件系统（Plugins）、当LangChain开始流行、当各种大模型都开始支持函数调用时，所有这些实现的思想根源都可以追溯到ReAct论文中阐述的Thought-Action-Observation循环。

从更深层的角度看，ReAct解决了一个语言模型领域的根本困境：即使是最强大的LLM，其知识也是被冻结的。模型在训练时知道什么，就只能在部署时知道什么。但现实世界是动态变化的——新闻在发生、股票在波动、代码在执行。ReAct提供的是一个架构，使得模型可以在运行时动态地与环境交互，获取最新的信息，甚至改变环境的状态。

从某种意义上说，ReAct是连接"纯推理模型"和"行动agent"的桥梁。Chain-of-Thought让模型学会了思考，而ReAct让模型学会了在思考的同时采取行动。

这个演进的重要性体现在后续研究的爆发式增长上。ReAct发表后的一年内，数十篇论文和数百个开源项目都基于或扩展了这个框架。HuggingFace的ReAct实现、LangChain的Agent系统、AutoGPT的设计——这些都是ReAct思想的不同演绎。

而对于实际应用者而言，ReAct带来的启示是：不要试图让一个模型记住所有信息，而应该教它如何去检索、调用和验证信息。这转变了我们对模型能力的理解——一个能够有效使用工具的小模型，有时候比一个知识量庞大但不能行动的大模型更有价值。

2023年和2024年，当Auto-GPT、GPT-4 Plugins、Claude的工具使用能力等相继推出时，它们的核心设计原理都已经在ReAct这篇论文中得到了充分的阐述。看似简单的Thought-Action-Observation循环，已经成为了重塑大语言模型应用方式的基础范式。