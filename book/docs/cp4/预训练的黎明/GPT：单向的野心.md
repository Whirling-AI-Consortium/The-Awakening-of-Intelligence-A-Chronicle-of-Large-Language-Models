---
sidebar_position: 1
---

## 相反的方向

在2018年的上半年，如果你在一个NLP学术会议上进行一次小范围的调查，问研究者们一个问题："预训练一个双向模型和预训练一个单向模型，哪个更有前景？"，绝大多数人的答案会毫不犹豫：双向。这个直觉是合理的。双向模型能看到完整的上下文，获得更多的信息，那为什么不用呢？

但OpenAI的一个年轻研究员Alec Radford，却在思考一个看似疯狂的反向问题：为什么一定要做双向的呢？

这个想法不是凭空产生的。Radford和他的团队在阅读了大量关于语言建模的文献后，意识到了一个被许多人忽视的事实：生成任务本质上是单向的。当你写一段文字时，你一次只能写一个词。前面的词已经决定了，你无法改变它们。你只能基于已经写下的内容，决定下一个词应该是什么。这不仅是计算上的现实，更是语言生成的本质。

而自回归建模（autoregressive modeling）——逐个生成下一个词的方式——自然适应了这种单向的因果结构。如果你用自回归模型来生成文本，给定前面所有的词，模型学会了预测下一个词。这个过程不需要"展望未来"。事实上，让模型能够看到它还没有生成的未来词，从因果关系上讲是不对的。

这就是GPT选择单向因果建模的核心逻辑。而这个选择，尽管与当时学术界的主流直觉相悖，却会在接下来的几年里证明自己是深谋远虑的。

## Transformer Decoder-Only的激进选择

Radford团队在设计GPT时面临的第一个架构决策是：用Transformer的哪一部分？

Transformer论文发表于2017年，它由两个对称的部分组成：编码器（Encoder）和解码器（Decoder）。编码器处理输入序列，通过自注意力生成隐表示。解码器处理输出序列，既能看到前面生成的输出（通过因果掩蔽），也能通过交叉注意力看到编码器的输出。这个结构对于机器翻译这样的序列到序列的任务非常合适——你需要理解源语言（编码器），然后生成目标语言（解码器）。

但对于语言建模，情况不同。语言建模本质上是一个单一序列的任务：给定前面的词，预测下一个词。你不需要编码和解码这样的二元结构。你只需要一个能够进行自回归生成的模块。

Radford的团队做出了一个激进的简化：他们丢弃了Transformer的编码器部分，只保留了解码器。更准确地说，他们保留了解码器的自注意力机制，但移除了编码器-解码器的交叉注意力。结果是一个纯粹的、单向的、自回归的架构。这个架构后来被称为Transformer Decoder-Only或Transformer-LM（Language Model）。

这个决定看似大胆，实际上却是极其聪慧的。一方面，它化繁为简，减少了架构的复杂性。另一方面，它直接针对语言建模这个任务进行了优化。不需要的交叉注意力被移除了，模型的所有注意力都用于捕捉同一序列内的依赖关系。

**关键的因果掩蔽机制**

Decoder-only架构的核心是因果掩蔽（causal masking）。在标准的自注意力计算中，每一个位置都能看到序列中所有其他位置。但在语言建模中，这是不允许的。一个词不应该能看到在它之后的词，因为那些词在生成时还没有产生。

因果掩蔽通过一个简单的技巧来实现：在计算注意力权重时，对于位置i，我们将所有j > i的注意力权重设置为-∞（或一个很大的负数），这样在softmax之后这些权重就会变为0。结果是，位置i只能注意到位置0到i的内容。

```python
# 因果掩蔽的实现原理
def causal_attention(query, key, value, seq_length):
    """
    query, key, value: [batch_size, seq_length, d_model]
    返回应用了因果掩蔽的注意力输出
    """
    
    # 计算注意力分数
    scores = matmul(query, transpose(key)) / sqrt(d_model)
    # scores shape: [batch_size, seq_length, seq_length]
    
    # 创建因果掩蔽矩阵
    # 这是一个下三角矩阵，只有i >= j的位置是1
    causal_mask = create_lower_triangular_mask(seq_length)
    # causal_mask shape: [seq_length, seq_length]
    
    # 应用掩蔽：将不应该看到的位置设为很小的值
    scores = where(causal_mask, scores, -1e9)
    
    # 计算注意力权重
    attention_weights = softmax(scores, dim=-1)
    # 此时，对于位置i，attention_weights[i, j>i] ≈ 0
    
    # 应用权重到value
    output = matmul(attention_weights, value)
    
    return output
```

这个看似简单的掩蔽机制，却有深远的意义。它在数学上强制了模型的因果结构，确保了自回归生成的可行性。

## 从117M到预训练的实验

2018年的OpenAI GPT（后来被称为GPT-1）使用了一个相对较小的模型：117M参数。在今天看来，这个规模已经不值一提，但在当时，这已经是一个可观的模型。

这个模型的预训练使用了一个包括约800万篇文章的数据集，称为BookCorpus。选择图书作为预训练语料是一个重要的设计决策。与互联网爬虫数据相比，图书往往质量更高、句子结构更复杂、语言更规范。这个选择影响了模型学到的语言特征。

预训练的目标非常直接：最大化语言建模的目标函数。给定一个句子中的前n-1个词，模型学习预测第n个词。这个过程非常古老——早可以追溯到计算机科学的早期。但在Transformer+大规模预训练的背景下，这个古老的想法突然展现出了新的力量。

```python
# GPT预训练的目标函数
def gpt_language_model_loss(model, batch_tokens, seq_length):
    """
    model: GPT模型
    batch_tokens: [batch_size, seq_length]
    seq_length: 最大序列长度（如512）
    """
    
    # 前向传播
    logits = model(batch_tokens[:, :-1])  # 输入所有词除了最后一个
    # logits shape: [batch_size, seq_length-1, vocab_size]
    
    # 目标是预测下一个词
    targets = batch_tokens[:, 1:]  # 所有词除了第一个
    # targets shape: [batch_size, seq_length-1]
    
    # 计算交叉熵损失
    loss = cross_entropy_loss(logits, targets)
    
    return loss
```

在这种简单而纯粹的预训练下，GPT学到了什么？论文的实验给出了惊人的答案。

## 微调和多任务能力

真正令人瞩目的地方在于微调阶段。Radford等人在预训练后，将GPT应用于多个下游NLP任务，并通过最少量的任务特定修改来进行微调。他们使用了一个几乎通用的微调框架，只在任务的输入/输出格式上做了必要的调整。

这是对当时NLP实践的一次颠覆。在2018年，标准的做法是为每个任务设计特定的架构。一个用于文本分类的模型和一个用于句子相似性任务的模型，会有明显不同的架构设计。但GPT团队展示的是，一个统一的预训练模型，配合最小化的任务适配层，就能在多个任务上取得竞争力的结果。

他们测试的9个任务包括：文本蕴含识别（RTE）、相似性检测（MRPC）、语义相似性（STS-B）、问答（QNLI）、问题对相似性（QQP）、视觉常识推理（VCR）、情感分析（SST）、语言可接受性（CoLA）和词汇相似性（MRPC等）。在这些任务中，GPT取得了接近或优于之前的SOTA（State-of-the-Art）结果。

更令人印象深刻的是，这些改进是通过一个极为简洁的微调策略取得的。论文的作者们甚至可以说，他们的方法在某种意义上是"无参数高效"的——大部分参数来自预训练，微调时只需要添加一个小的任务特定的线性层。

## 与BERT的对比：生成vs理解的分歧

当GPT论文发表几个月后，Google的BERT论文闪亮登场时，学术界突然意识到自己站在了一个十字路口。两个方向，都有雄厚的理论基础和实验支持。

BERT采用了ELMo所启发的双向预训练方向，但用Transformer Encoder代替了LSTM。它的核心思想是：通过遮蔽一些词，让模型学习从上下文中恢复这些词。这是一个完形填空任务，本质上是一个理解任务。BERT的模型架构是Transformer Encoder-Only——它没有因果结构，完全可以双向地看。

GPT则坚持了生成方向。它的预训练目标是标准的自回归语言建模，模型通过因果掩蔽只能单向地看。在架构上，GPT使用的是Transformer Decoder（带因果掩蔽）。

这个差异看似技术性的，实际上代表了两种不同的哲学：

**BERT的哲学**：语言理解是首要的。如果你能够真正理解一个句子（通过双向上下文），那么你可以处理任何依赖于理解的任务。生成可以看作理解的一个应用。因此，BERT将预训练的资源集中在理解任务上。

**GPT的哲学**：生成是最基本的。语言建模——预测下一个词——是语言学习的最纯粹的形式。它不需要显式的任务定义，可以在任何原始文本上进行。而且，如果你学会了生成，你实际上也学会了理解（因为生成过程隐含了对语言结构和语义的理解）。

实际上，从模型能力的角度看，两个哲学都有其合理性。BERT在纯理解任务（如文本分类、句子配对）上表现优异，因为它的双向结构直接对应这些任务的需求。而GPT在生成和开放式任务上有天然的优势。

但在2018年底到2019年初的时间点上，业界普遍认为BERT会是赢家。理由很充分：BERT在主流的NLP基准上的表现更好，而生成任务相对来说是一个小众的应用。几乎没人预测到，五年后，生成任务会成为AI界最热门的应用。

## Alec Radford的远见与OpenAI的策略

但OpenAI似乎坚持了一种长期的视角。Alec Radford和他的团队在2018年末透露，他们已经在训练一个更大的GPT模型——参数量达到1.5B。这个模型会在一个更大的数据集（WebText）上训练，后来会以GPT-2的名义发布。

在GPT-1的论文中，Radford等人虽然看到了他们模型的多任务能力，但似乎还没有完全意识到单向因果建模会有多大的潜力。那个时刻，他们的话语中充满了谨慎和好奇，而不是确定和宣言。

但有一个细节值得注意。在论文中，作者们提到，他们观察到，随着预训练数据的增加，模型在下游任务上的性能也在改善。这个观察，在当时相对来说不是新闻，但它为后来的Scaling Laws埋下了种子。

而且，Radford的选择有一个深层的原因。由于GPT采用了单向因果建模，它天然适应于文本生成。这意味着，一旦模型足够大和足够好，它就能直接被用于生成任务，而不需要复杂的微调或任务特定的修改。这是一个强大的特性，即使在当时还没有被充分利用。

## 代码实现与细节

要理解GPT为什么有效，最好的方法是看一个简化的实现：

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        
        # Decoder-only堆栈
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_length]
        返回logits: [batch_size, seq_length, vocab_size]
        """
        # 嵌入和位置编码
        x = self.embedding(input_ids)  # [batch_size, seq_length, d_model]
        x = x + self.pos_embedding(input_ids)
        
        # 创建因果掩蔽
        causal_mask = create_causal_mask(input_ids.shape[1])
        
        # 通过decoder层
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, causal_mask=causal_mask)
        
        # 输出层到logits
        logits = self.lm_head(x)  # [batch_size, seq_length, vocab_size]
        
        return logits

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, causal_mask):
        # 自注意力（带因果掩蔽）
        attn_output = self.self_attention(x, mask=causal_mask)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

# 训练循环
def train_step(model, batch_tokens, optimizer):
    # Forward pass
    logits = model(batch_tokens[:, :-1])
    
    # 计算损失（预测下一个词）
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        batch_tokens[:, 1:].contiguous().view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

这个实现的简洁性本身就说明了Decoder-only架构的优雅。没有编码器-解码器的复杂性，没有交叉注意力的额外复杂度。只有单纯的、因果掩蔽的自注意力，和一个标准的前馈网络。

## 对后续发展的影响

虽然在2018-2019年，学术界和业界似乎更看好BERT这个方向，但GPT其实已经为未来的发展打下了伏笔。

首先，Decoder-only的简洁性意味着它可以更容易地扩展到更大的规模。没有编码器-解码器的不对称性，没有需要对齐的两个部分。只要堆积更多的层和参数，就能得到一个更大的模型。

其次，因果建模和自回归生成的直接对应关系，意味着GPT天然适应于在无标注数据上进行大规模预训练。你不需要任何标注，只需要原始文本。而BERT虽然也不需要标注，但它的双向结构和完形填空目标，在大规模生成任务中就变成了一个限制因素。

第三，也是最重要的一点，GPT选择的这个方向——纯粹的、无条件的、自回归的语言建模——它的上限是什么？没人知道。这种未知性，对于研究者来说是吸引人的。也许，这就是真正通向通用人工智能的道路。

当后来的GPT-2（2019）、GPT-3（2020）相继发布，并展现出越来越惊人的能力时，人们才开始意识到，Alec Radford在2018年的选择，也许不是一个激进的赌博，而是一种深邃的先见。单向的野心，也许正是指向多能力未来的灯塔。

## 开放的问题

GPT-1时代遗留给我们的问题，比BERT时代的问题更具有开放性。首先，纯粹的自回归建模是否真的足够？它能否通过纯粹的参数增加来完成越来越复杂的任务？其次，Decoder-only架构的极限在哪里？是否存在某个模型大小或数据量，之后性能开始下降？第三，也许最有趣的：一个在自回归语言建模上预训练的模型，真的能够学到足够的"理解"来处理理解型任务吗？

这些问题，到GPT-2的时代仍然没有明确的答案。但那个时代，答案开始变得有趣起来。