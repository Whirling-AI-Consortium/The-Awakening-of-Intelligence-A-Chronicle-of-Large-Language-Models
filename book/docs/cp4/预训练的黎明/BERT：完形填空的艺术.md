---
sidebar_position: 2
---


## Google的王炸时刻

2018年10月，当Jacob Devlin和他在Google Brain的团队在arxiv上发表《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》时，没有人意识到这会成为NLP历史上最具影响力的论文之一。不是因为技术有多新颖——事实上，BERT的每一个组件都已经存在。而是因为它找到了一个如此简单、如此有效、如此优雅的方式，来利用这些已有的技术。

这篇论文发表的时机非常微妙。ELMo已经证明了预训练的价值。GPT已经展示了Transformer可以用于语言建模。但业界对最优的预训练目标仍然没有共识。双向还是单向？完形填空还是自回归？在这个分水岭的时刻，Google交出了他们的答卷。

这个答卷的核心思想简单到几乎让人觉得有些荒谬：随机遮蔽15%的单词，然后让模型猜测它们是什么。就这样。没有复杂的目标设计，没有精妙的损失函数，只有一个小学生都能理解的游戏规则。

但是，有时候，最强大的想法正是最简单的那些。

## Masked Language Model：遮蔽的天才设计

MLM（Masked Language Model）的核心思想源自一个古老的教学技巧。当你给一个学生一篇有部分词汇缺失的文章，让他根据上下文填空时，这不仅是一个语言理解的测试，更是一个极其高效的学习过程。学生必须理解周围词汇的语义，才能正确地填空。这种理解必须是双向的——前文和后文的信息都很关键。

BERT的设计者们意识到，这个古老的教学技巧可以完美地转化为一个深度学习的预训练目标。而且，由于BERT有意地遮蔽了部分输入，它就不会"作弊"地通过简单的顺序预测来完成任务。它必须真正理解语言。

但实现MLM并不是简单地把词删除。Devlin等人设计了一个更精妙的方案：

**三步遮蔽策略**

当选定要遮蔽的15%的词时，对于这些词：
- 80%的时候，用特殊的[MASK]标记替换（例如"The cat sat on the [MASK]"）
- 10%的时候，用一个随机的词替换（例如"The cat sat on the dog"，虽然这在语义上毫无意义）
- 10%的时候，保持原词不变（例如"The cat sat on the mat"）

这个看起来奇怪的设计有其深刻的理由。如果总是用[MASK]替换，模型在实际应用时会看到原始文本中没有[MASK]标记，可能导致分布转移。通过在20%的时候保持原词或随机替换，模型被迫学习在每个位置都预测原始词，而不是依赖[MASK]标记的出现来"知道"这里应该做预测。这强制了模型学习更深层的表示。

```python
def masked_language_model_objective(tokens, vocab_size, mask_prob=0.15):
    """
    实现MLM的目标函数
    tokens: 原始序列 [batch_size, seq_length]
    """
    
    batch_size, seq_length = tokens.shape
    
    # 决定哪些位置要进行MLM
    mlm_positions = []
    mlm_labels = []
    modified_tokens = tokens.clone()
    
    for i in range(batch_size):
        for j in range(seq_length):
            if tokens[i, j] == PAD_ID:  # 跳过padding
                continue
            
            # 以mask_prob的概率选择这个位置
            if random.random() < mask_prob:
                mlm_positions.append((i, j))
                mlm_labels.append(tokens[i, j])
                
                # 三步策略
                rand = random.random()
                if rand < 0.8:
                    # 80%: 用[MASK]替换
                    modified_tokens[i, j] = MASK_ID
                elif rand < 0.9:
                    # 10%: 用随机词替换
                    modified_tokens[i, j] = random.randint(0, vocab_size - 1)
                # 10%: 保持原词不变
    
    # Forward pass with modified tokens
    logits = model(modified_tokens)
    
    # 计算损失：只在MLM位置计算
    mlm_loss = 0
    for (batch_idx, seq_idx), true_label in zip(mlm_positions, mlm_labels):
        pred_logits = logits[batch_idx, seq_idx, :]
        mlm_loss += cross_entropy(pred_logits, true_label)
    
    return mlm_loss / len(mlm_positions)
```

这个三步策略虽然复杂，但也正是为什么MLM如此有效。它防止了模型过度适应[MASK]标记这个虚拟的符号。

## Next Sentence Prediction：被遗忘的部分

除了MLM，BERT的预训练还包括另一个目标：NSP（Next Sentence Prediction）。给定两个句子A和B，模型需要预测B是否是在文本中紧跟在A后面的真实句子，还是从语料库中随机选择的句子。

```python
def next_sentence_prediction_objective(sentences_pair, model):
    """
    NSP目标函数
    sentences_pair: [(sent_A, sent_B, is_next), ...]
    is_next: 布尔值，B是否真的跟在A后面
    """
    
    losses = []
    
    for sent_a, sent_b, is_next in sentences_pair:
        # 构造输入：[CLS] sent_a [SEP] sent_b [SEP]
        input_tokens = [CLS] + sent_a + [SEP] + sent_b + [SEP]
        
        # Forward pass
        cls_output = model(input_tokens)[0]  # 取[CLS]对应的输出
        
        # NSP分类器：二元分类
        nsp_logits = nsp_classifier(cls_output)  # 输出维度为2
        
        # 计算损失
        nsp_loss = cross_entropy(nsp_logits, int(is_next))
        losses.append(nsp_loss)
    
    return mean(losses)
```

NSP的目标是为模型引入对"句子关系"的理解。这在一些任务中是有用的，比如句子配对任务。但后来的研究发现，NSP的贡献其实没有MLM那么大，甚至在某些情况下可能是多余的。后续的模型（如RoBERTa）会移除NSP，而只保留MLM。但在BERT的原始设计中，这两个目标是并行的。

## 双向Transformer Encoder的威力

BERT使用的模型架构是Transformer的Encoder部分，没有因果掩蔽。这意味着，在计算自注意力时，每个位置都能看到序列中的所有其他位置。

这与GPT的Decoder-only架构形成了鲜明对比。GPT因为要生成文本，所以需要因果掩蔽来防止模型看到未来的词。但BERT在预训练阶段不需要生成，只需要理解。所以它可以充分利用双向的信息流。

这个架构选择有两层意义。首先，从计算效率的角度，双向的信息流比单向更稠密。每个词都能与整个序列的所有词交互，而不仅仅是前面的词。这通常导致更丰富的表示学习。其次，从任务匹配的角度，大多数NLP任务（分类、配对、标注）都是"理解"型的，不需要生成。所以双向模型直接对应了这些任务的需求。

```python
class BertEncoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)  # 句子A或B
        
        # Encoder堆栈（无因果掩蔽）
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, token_type_ids):
        """
        input_ids: [batch_size, seq_length]
        token_type_ids: [batch_size, seq_length] (0 for sent_A, 1 for sent_B)
        """
        # 嵌入层
        x = self.embedding(input_ids)
        x += self.pos_embedding(input_ids)
        x += self.token_type_embedding(token_type_ids)
        
        # 创建attention mask（无因果约束）
        attention_mask = create_attention_mask(input_ids)  # 仅mask padding
        
        # 通过encoder层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask=attention_mask)
            # 注意：没有因果掩蔽，每个位置可以看到所有位置
        
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attention_mask):
        # 自注意力（无因果掩蔽，完全双向）
        attn_output = self.self_attention(x, mask=attention_mask)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
```

值得注意的是，BERT还在嵌入中加入了token type embedding。这用来区分输入中的两个句子（如果有的话）。对于单句输入的任务，所有token的type_id都是0。对于句子对的任务，前半部分是0，后半部分是1。这个设计虽然简单，但在处理多句输入时非常有用。

## 横扫11项任务的王者

当BERT发表时，Google附加了一系列惊人的实验结果。在GLUE基准（General Language Understanding Evaluation）上，BERT在9个任务中都达到了SOTA。在SQuAD v1.1和v2.0问答基准上，BERT也超越了之前的最强结果。总计，论文声称在11项不同的NLP任务上刷新了记录。

这不仅仅是一个数字上的胜利。这代表了一个范式的转变。在BERT之前，NLP的做法通常是：对于每个特定任务，设计一个特定的架构，预训练一个特定的模型（如果需要的话），然后微调。结果是一个高度碎片化的生态——需要一个不同的系统来处理分类、一个不同的系统来处理序列标注、另一个不同的系统来处理问答。

BERT改变了这一切。一个统一的预训练模型，配上最小化的任务特定调整，就能在所有这些任务上取得最佳结果。这不仅简化了工程，也有深刻的科学含义：也许，存在一个通用的、与任务无关的"语言理解"能力，而BERT学到了它。

**BERT的微调策略**

对于不同的下游任务，BERT的微调方式是一致的：

1. **分类任务**（如情感分析）：取[CLS]标记（序列开头的特殊标记）对应的输出，通过一个线性分类器预测类别。

2. **标注任务**（如NER）：取每个token对应的输出，通过一个线性层和softmax预测该token的标签。

3. **配对任务**（如句子相似性）：输入格式为[CLS] sent_a [SEP] sent_b [SEP]，取[CLS]的输出进行二分类或回归。

4. **问答任务**：输入格式为[CLS] question [SEP] passage [SEP]，预测在passage中的start和end位置。

这种统一性是Transformer架构相对于之前RNN方法的一个主要优势。由于Transformer是完全非递归的，它可以灵活地处理各种长度和结构的输入。

## 论文的科学洞察

除了实验结果，BERT论文还进行了深入的分析实验。其中最有启发性的是对不同层的表示的分析。

论文的作者们进行了一个关键实验：他们取了BERT在不同层的隐状态，使用这些层的表示来解决具体的任务，而不使用最后一层。结果表明：

- 低层（接近输入）的表示更多地捕捉浅层的句法信息。
- 中层的表示开始包含任务特定的信息。
- 高层（接近输出）的表示最有用，但不同任务的最优层数不同。

这个发现意味着，BERT学到的是一个分层的表示——从具体的语法特征到抽象的语义特征，再到任务相关的信息。这与人类语言处理的多层次本质相吻合。

```python
def analyze_layer_contributions(model, task_tokens, task_labels, num_layers):
    """
    分析不同层对任务的贡献
    """
    
    # 获取所有层的输出
    all_layer_outputs = []
    for layer_idx in range(num_layers):
        layer_output = model.get_layer_output(task_tokens, layer_idx)
        all_layer_outputs.append(layer_output)
    
    # 对每一层单独进行微调和评估
    layer_performance = []
    for layer_idx, layer_output in enumerate(all_layer_outputs):
        # 在该层之上加一个简单分类器
        classifier = SimpleLinearClassifier(layer_output.shape[-1])
        
        # 微调并评估
        acc = train_and_eval(classifier, layer_output, task_labels)
        layer_performance.append(acc)
    
    return layer_performance
    # 结果通常显示：
    # - 第0层（输入）：较低的性能
    # - 中间层：逐步提升
    # - 最后一层：最高性能
    # - 特殊情况：某些语法任务在中层表现更好
```

## 完形填空的哲学

BERT的核心设计选择——完形填空——背后有一个深刻的哲学。这个设计反映了对"理解"的一种特定的认识。

完形填空测试的是什么？它测试的是你能否根据上下文推断一个缺失的单词。这需要：

1. **语法理解**：你需要知道这个位置应该是什么词性、什么时态。
2. **语义理解**：你需要理解句子的含义，推断出逻辑一致的词。
3. **常识推理**：在某些情况下，正确的填空需要对世界的理解。

而且，完形填空是对称的、双向的。无论你看哪个方向的上下文，你都在做同样的推断任务。这与一个"真正的理解"应该具有的性质相符——理解不应该因为信息来自前还是后而改变。

相比之下，GPT的自回归语言建模只要求单向的推断——从前文推断下一个词。这对于生成任务是最优的，但对于"理解"可能不够全面。

BERT的设计者们似乎在说：如果你想要一个真正"理解"语言的模型，就让它做一个理解型的任务——完形填空。

## 对预训练范式的深远影响

BERT的发表是2018年NLP的一个分水岭。它不仅仅是一个更好的模型或一套新的技术。它定义了一个新的范式：预训练-微调。

在BERT之前，虽然ELMo已经展示了预训练的价值，但预训练仍然被看作是可选的、是"调优"的一部分。大多数研究者仍然习惯于从头开始训练任务特定的模型。

BERT改变了这一切。BERT表明，预训练不仅有用，而且是必须的。如果你不使用预训练模型，你的结果会明显更差。从BERT发表之后，预训练-微调范式成为了NLP的标准。没有预训练，你在2018年之后就被看作是在做"旧式的"NLP。

这个范式转变的影响是深远的。它改变了整个行业的工作流程。不再需要为每个任务单独训练模型。不再需要对每个任务做细致的架构设计。只需要：下载一个预训练模型，在你的数据上微调，完成。

这个转变也加速了"大模型"时代的到来。既然预训练是关键，那么在预训练上投入更多资源就变得有理。这直接推动了模型规模的增长——从BERT的3.4B参数（base版本）到后续模型的更大规模。

## 开放的问题与批评

尽管BERT取得了巨大的成功，但它也留下了一些问题。首先，MLM和NSP这两个目标中，哪一个更重要？后续研究（如RoBERTa）表明，NSP可能是不必要的，甚至是有害的。这提出了一个问题：最优的预训练目标应该包含什么？

其次，15%的遮蔽比例是最优的吗？为什么不是20%或10%？BERT论文中虽然进行了一些消融实验，但对这个超参数的选择的理论理由并不充分。

第三，完形填空真的是理解的最佳模型吗？一些批评者指出，完形填空与下游任务（如分类）之间仍然有显著的分布差异。一个在完形填空上表现完美的模型，可能在分类上的性能依然有限。

最后，也许最深刻的问题是：BERT学到的"理解"的极限是什么？当模型扩展到更大规模时，这种理解会如何发展？答案会在后续的GPT-3、DALL-E等大规模模型上逐渐揭露。

但在2018年的那个时刻，当《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》这篇论文发表时，它定义了一个新的纪元。完形填空这个古老的教学技巧，通过深度学习的方式，成为了理解语言最有效的方法之一。这就是BERT的艺术。