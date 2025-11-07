---
sidebar_position: 1
---


## 不能承受的复杂度之重

Transformer的Attention机制是一个计算密集的操作。当处理长度为$n$的序列时，每个token需要与序列中的所有其他token计算相似度，这导致了$O(n^2)$的时间复杂度和$O(n^2)$的空间复杂度。在模型刚刚诞生的2017年，这并不是一个紧迫的问题——研究者们关心的是512或1024长度的序列。但当我们跨入大模型时代，这个二次复杂度逐渐演变为一个梦魇。

考虑一个简单的数字：一个支持4K tokens的模型，其Attention矩阵大小为4096×4096，需要1600万个浮点数。如果要支持64K tokens，这个数字会膨胀到40亿。即使使用Float16精度，单个Attention矩阵也需要数GB的显存。而且这仅仅是内存，计算时间同样以$n^2$增长——从4K到64K，计算时间会增加256倍。到了2022年中期，当研究者们试图训练或微调支持32K上下文的模型时，显存溢出和训练超时成为了普遍的痛点。

这个问题有一个深层的假设：每个token都需要关注序列中的每一个其他token。但这真的必要吗？

## 局部性假设的威力

人类的注意力显然不是全局的。当我们阅读一份长文档时，眼睛通常只在邻近的句子间游移，而不是逐字向后翻阅整个上下文。这个直观观察背后是一个深刻的认知学原理：信息的相关性往往与距离成反比。你的大脑不需要关注整个文档来理解当前的句子，邻近的上下文已经提供了大部分必要的信息。

这个洞察驱动了稀疏注意力（Sparse Attention）的诞生。2020年初，Google的Nikita Kitaev等人在论文《Reformer: The Efficient Transformer》中提出了一个革命性的想法：与其计算完整的$n \times n$ Attention矩阵，不如限制每个token只与邻近的token进行Attention计算。他们称这种设计为局部注意力（Local Attention）。

局部注意力的实现很直接：对于位置$i$的token，只计算与范围$[i-w, i+w]$内token的Attention权重，其中$w$是窗口大小。这将复杂度从$O(n^2)$降低到$O(n \cdot w)$。如果窗口大小固定为常数（比如512），那么总复杂度就变成了$O(n)$——一个质的飞跃。

但仅有局部信息是不够的。考虑这样一个场景：你在阅读一篇论文的结论部分，而关键的定义出现在五页之前。纯粹的局部注意力会导致模型无法回溯到远处的重要信息。这就产生了稀疏注意力设计中的一个经典权衡：如何在计算效率和信息流通之间找到平衡？

## 滑动窗口的实用方案

在2023年初，Mistral AI发布了一个参数量仅为7B但性能媲美13B模型的语言模型。这个奇迹的背后，就是他们对Sliding Window Attention的巧妙运用。Mistral团队的思路很务实：不追求最优的理论设计，而是采取一个简单有效的工程方案。

Sliding Window Attention的核心机制是：每个token只与一个固定大小的窗口内的token进行Attention计算。举例来说，在Mistral 7B中，窗口大小为4096。这意味着当处理第10000个token时，它只需关注第6000到第10000个token之间的相互关系。这个设计的妙处在于它的极端简洁性——没有复杂的路由机制，没有可学习的参数，仅仅是对Attention掩码的一个简单修改。

```python
def create_sliding_window_mask(seq_len, window_size):
    # 创建滑动窗口注意力掩码
    # (seq_len, seq_len) 的布尔矩阵，True表示可以注意
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    
    for i in range(seq_len):
        # 每个位置i只能关注 [max(0, i-window_size), i] 范围
        start = max(0, i - window_size)
        mask[i, start:i+1] = True
    
    return mask

# 在Attention计算中应用掩码
def scaled_dot_product_attention_windowed(Q, K, V, window_size):
    # Q, K, V: (batch, seq_len, dim)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
    
    # 创建并应用窗口掩码
    mask = create_sliding_window_mask(Q.shape[1], window_size)
    scores = scores.masked_fill(~mask, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)
```

Sliding Window Attention的优势显而易见。首先，它能被高效地实现在现有的硬件上。标准的GPU Kernel可以很轻松地支持这种结构化的稀疏模式。其次，它的灵活性很高——窗口大小可以在不重新训练模型的情况下动态调整。再次，从实验结果来看，局部注意力保留了模型对邻近上下文的精细建模能力，在大多数NLP任务上性能损失微乎其微。

然而，Sliding Window Attention也有其局限。如果关键信息距离当前位置超过窗口大小，模型就会彻底忽视它。在处理具有长期依赖的任务时（如检索长文档中的特定事实），这个限制可能成为瓶颈。

## 稀疏注意力的多种变体

Sliding Window不是唯一的稀疏注意力方案。研究者们在过去几年探索了各种创意性的模式。

**Longformer**（2020年，Allen AI）采用了一个分层的方案。它将Attention分为两部分：局部窗口注意力和全局注意力。某些被标记为"全局"的token（比如[CLS] token或特定的关键词）可以关注整个序列，而普通token只能进行局部注意力。这个设计的妙处在于用少量的全局token充当信息的"路由节点"，确保重要信息能够流通。论文《Longformer: The Long-Document Transformer》展示了这个方案在长文档分类和问答任务上的有效性。

```python
def longformer_attention(Q, K, V, window_size, global_token_indices):
    seq_len = Q.shape[1]
    scores = torch.full((seq_len, seq_len), float('-inf'))
    
    # 1. 局部注意力：每个token关注窗口内的token
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        scores[i, start:end] = compute_scores(Q[i], K[start:end])
    
    # 2. 全局注意力：全局token关注所有token，所有token都关注全局token
    for idx in global_token_indices:
        scores[idx, :] = compute_scores(Q[idx], K)  # 全局token看整个序列
        scores[:, idx] = compute_scores(Q, K[idx])  # 所有token都看全局token
    
    return scores
```

**BigBird**（Google, 2020）则采用了随机稀疏的方案。除了局部窗口和全局token，BigBird还添加了随机选择的token用于Attention计算。这个想法源于计算机科学中的随机图论——用随机连接确保消息能够快速传播。虽然看起来有点"赌博"的味道，但实验证明随机稀疏度对模型的泛化能力反而有所帮助。

**Reformer**（Google, 2020）则采用了局部敏感哈希（Locality-Sensitive Hashing, LSH）的想法。它不是按固定位置创建窗口，而是根据向量相似度将相似的token分组，然后只在组内计算Attention。这个方案在理论上很优雅——相似的token无论位置多远都能相互看见——但在实践中，哈希冲突和分组的不稳定性给训练带来了挑战。

## 后顾与前瞻

到了2023年，滑动窗口注意力已成为支持长上下文的标配方案。除了Mistral的开源贡献，Meta的Llama 2在某些版本中也采用了类似的局部注意力机制。Anthropic在Claude中则采用了一个更复杂的混合方案——结合了滑动窗口、全局token和其他优化技术。

有趣的是，当我们观察大模型在实际应用中的表现时，会发现一个反直觉的现象：相比完整的全局注意力，适当的稀疏注意力有时反而能提升模型的泛化性能。论文《The Curious Case of Language Generation Evaluation Metrics》和后续的研究表明，这可能源于一个微妙的正则化效应——限制注意力范围反而能防止模型过度拟合特定的上下文模式。

然而，纯粹的局部注意力仍有其局限。某些任务如长文档级别的信息检索，需要模型回溯到几千个token之前的段落。为此，研究者们在2023年开始探索更复杂的混合方案。一些最新的论文提出了条件稀疏注意力（Conditional Sparse Attention）——模型根据输入动态决定应该使用全局注意力还是局部注意力。还有研究尝试了多个注意力头的差异化设计：某些头采用局部注意力以获得高效率，某些头保持全局注意力以保留长期依赖。

令人兴奋的是，这些努力正在打破我们对Attention复杂度的刻板认识。当Gemini 1.5声称支持百万token时，其背后很可能采用了某种多层级的稀疏注意力机制——或许是基于注意力熵的动态稀疏化，或许是基于文档结构（段落、章节等）的层级式注意力。具体细节尚未公开，但技术社区普遍认为，完全密集的全局注意力已经不再是支持超长上下文的必经之路。

稀疏注意力的演进本质上反映了一个深刻的计算学原理：全局优化往往不如局部启发式加以适当的冗余。Sliding Window的成功在于它承认了一个现实——绝大多数相关性是局部的，而那些罕见的长距离依赖可以通过其他机制（如检索或显式的上下文管理）来处理。这个转变标志着我们从追求理论上的完美性，转向了实用主义的工程智慧。