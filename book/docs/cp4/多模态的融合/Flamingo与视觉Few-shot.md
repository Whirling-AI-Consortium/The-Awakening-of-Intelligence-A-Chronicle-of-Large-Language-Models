---
sidebar_position: 1
---


## 大模型时代的多模态困境

2021年底，当OpenAI发布CLIP后，整个研究社区面临了一个新的困境。CLIP证明了图像和文本可以在同一个嵌入空间中对齐，这为多模态学习打开了一扇新的大门。然而，随之而来的问题同样紧迫：如何把这个强大的视觉编码器与语言生成能力结合在一起？

最直接的方式是一个朴素的想法——既然CLIP能理解图像，既然GPT-3能理解语言并生成文本，那为什么不直接把两个模型拼接在一起呢？把CLIP的图像表示丢进GPT-3，让它生成描述。理论上，这应该能工作。但实践中，这样的拼接往往效果不佳。问题在于，CLIP和GPT-3在预训练时学到的表示空间完全不同，它们之间缺乏深度的对齐。而且，两个冻结的预训练模型之间的简单连接无法有效地进行信息流动。

更深层的问题来自预训练范式本身。当时，大多数视觉语言模型都采用一种天真的方式：冻结一个预训练的视觉编码器，再加上一个语言模型，然后在有标注的图像-文本对数据集上进行微调。这种方式有两个严重的缺陷。首先，没有大规模的、高质量的视觉-语言配对数据集可以用于微调，而互联网上充满的是"野生"的、未标注的图像-文本对。其次，即使能找到标注数据，微调预训练模型也需要大量的计算资源，这对于一个70B参数的模型来说是天文数字。

2022年4月，DeepMind的Jean-Baptiste Alayrac等人发布了论文《Flamingo: a Visual Language Model for Few-Shot Learning》。与其说这是一个关于新架构的论文，不如说这是一篇关于"如何优雅地连接两个预训练模型"的设计哲学研究。Flamingo是一个单一的视觉语言模型，在大范围的开放式多模态任务的少样本学习中建立了新的技术水平。Flamingo的核心思想简洁而深刻：不是通过微调来改变预训练模型，而是通过精心设计的桥接层和交叉注意力机制，让两个冻结的预训练模型能够协同工作。这个思想后来影响了GPT-4V、Gemini等一系列多模态大模型，成为了现代多模态学习的标准范式。

Flamingo之所以被称为"Flamingo"，可能是因为这只美丽的鸟类有着修长的脖子——而Flamingo的架构中那灵活的、贯穿整个模型的交叉注意力机制，就像这修长的脖子一样，优雅地连接了两个看似独立的部分。

## 架构创新：三层解耦设计

传统的多模态模型要么采用紧密耦合的架构（从头开始训练），要么采用简单的拼接方式（效果不佳）。Flamingo的创新在于提出了一个优雅的中间方案：在冻结的视觉编码器和冻结的语言模型之间，插入一个可训练的适配层。更精妙的是，这个适配层不是简单的投影或MLP，而是一个精心设计的多层次系统。

Flamingo的架构可以分为三个清晰的部分。第一部分是视觉编码器，使用的是一个冻结的、经过训练的Normalizer-Free ResNet。这个编码器从图像或视频中提取特征，产生一个空间特征网格。对于视频，模型每秒采样一帧，对每一帧独立编码，然后加上可学习的时间位置编码，将这个4D的时空张量展平成一个特征序列。

第二部分是Perceiver Resampler，这是整个架构中最创新的组件。传统的Perceiver模型由Jaegle等人在2021年提出，它通过一个固定的学习查询集合来处理任意长度的输入序列，将其重新采样为固定大小的潜在表示。Perceiver Resampler使用潜在向量和视觉编码器的输出，通过多个注意力层堆栈来提取来自图像序列的信息。

为什么需要这样的重新采样？原因很实际。视觉编码器输出的特征维度是可变的，这取决于输入图像的分辨率、输入的图像数量等因素。而下游的语言模型需要固定大小的输入。Perceiver Resampler优雅地解决了这个问题：无论输入多少张图像，无论分辨率如何，输出总是一个固定大小的张量。这意味着Flamingo可以灵活地处理单图、多图，甚至视频序列，而无需改变语言模型的架构。

第三部分是语言生成模块，使用的是一个冻结的、经过大规模预训练的Transformer解码器。Flamingo在Chinchilla模型的基础上构建，形成了一个80B参数的视觉语言模型。关键的创新在于，这个模型不是直接对语言模型进行微调，而是在其内部插入专门的门控交叉注意力块，与原始的预训练语言模型块交错。

## 门控交叉注意力：在冻结模型中注入新知识

门控交叉注意力（Gated Cross-Attention）是Flamingo突破的关键。在标准的Transformer解码器中，每一层都包含自注意力来处理文本之间的依赖关系。Flamingo的做法是，在这些原始的Transformer块之间定期插入新的交叉注意力块。这些交叉注意力块从Perceiver Resampler产生的"视觉token"中注入视觉信息。

但这里有一个微妙而重要的设计细节。如果简单地在冻结的预训练模型中插入新的层，会有一个问题：在初始化时，新层还没有学到任何有用的东西，它们会产生噪音，可能会破坏预训练模型的性能。为了解决这个问题，论文采用了一个优雅的技巧——tanh门控机制，它将新添加层的输出乘以tanh(α)，其中α是一个层特定的可学习标量，初始化为0。

具体来说，每个新插入的交叉注意力块的输出都会通过一个门控单元进行调制：

$$\text{output} = x + \tanh(\alpha) \cdot \text{GatedCrossAttn}(x, \text{visual\_tokens})$$

其中x是来自下层的输入，初始化时α = 0，所以tanh(0) = 0，新层的输出在初始化时为零。这意味着在训练开始时，整个模型的行为与原始的语言模型完全相同。随着训练进行，α逐渐增加，交叉注意力层的贡献逐步增强。这种渐进式的信息注入方式不仅保证了训练的稳定性，而且让模型能够平衡地融合视觉和语言信息。

交叉注意力的计算过程中有一个巧妙的设计：图像因果性。在处理文本序列时，当前的文本token只能看到在它之前出现的图像的视觉token，而不是所有之前的图像，这是通过对完整的文本到图像交叉注意力矩阵进行掩蔽来实现的。这个约束保证了模型能够学到有意义的依赖关系，避免"作弊"通过查看后续的图像来预测当前的文本。

对于多图像的情况，模型采用了一个简化的策略：每个文本token只直接参加到它相邻的那张图像的交叉注意力中，而不是所有之前的图像。但由于文本的自注意力和隐藏状态的传播，模型能够间接地获得所有之前图像的信息。这个单图像交叉注意力方案重要的是，它使得模型能够无缝地泛化到任意数量的视觉输入。

## 实现细节与交错式处理

Flamingo处理交错的视觉和文本输入的方式很有启发性。从数据格式的角度来看，输入是这样的：

```
<image_1> Text describing image 1. <image_2> Text describing image 2.
Question about both images? <o>
```

在内部，模型的处理流程如下所示（伪代码）：

```python
# 输入预处理
def preprocess_interleaved_input(text, images):
    # 文本通过tokenizer处理
    tokens = tokenizer.encode(text)
    
    # 找到<image>标记的位置
    image_positions = find_image_positions(tokens)
    
    # 图像通过视觉编码器和Perceiver Resampler处理
    visual_tokens_list = []
    for img in images:
        # 使用冻结的NFNet视觉编码器
        vision_features = vision_encoder(img)  
        # 压缩为固定大小的潜在表示
        visual_tokens = perceiver_resampler(vision_features)  
        visual_tokens_list.append(visual_tokens)
    
    return tokens, image_positions, visual_tokens_list

# 语言模型前向传播
def flamingo_forward(tokens, image_positions, visual_tokens_list):
    # 文本token嵌入
    hidden_states = embedding_layer(tokens)  # [seq_len, d_model]
    
    visual_token_idx = 0
    for layer_idx, (transformer_block, cross_attn_block) in enumerate(zip(
        pretrained_lm_blocks, gated_cross_attn_blocks)):
        
        # 通过预训练的Transformer块（完全冻结）
        hidden_states = transformer_block(hidden_states)
        
        # 在有图像的位置注入视觉信息
        for image_pos in image_positions:
            visual_tokens = visual_tokens_list[visual_token_idx]
            
            # 门控交叉注意力
            query = hidden_states[image_pos]  # 来自文本的查询
            cross_attn_output = cross_attn_block(
                query,
                visual_tokens,              # key/value来自视觉
                mask=image_causal_mask      # 应用图像因果性掩蔽
            )
            
            # 门控机制：初始时为零，逐渐增加
            gate_value = torch.tanh(alpha[layer_idx])
            # 残差连接与门控相结合
            hidden_states[image_pos] = hidden_states[image_pos] + \
                                        gate_value * cross_attn_output
        
        visual_token_idx += 1
    
    return hidden_states

# 文本生成
def generate_caption(image, prompt="A photo of"):
    # 一次性编码图像
    vision_features = vision_encoder(image)
    visual_tokens = perceiver_resampler(vision_features)
    
    # 初始化生成序列
    generated = tokenizer.encode(prompt)
    
    # 自回归生成
    for _ in range(max_length):
        # 前向传播，注入视觉信息
        logits = flamingo_forward(generated, [visual_tokens])
        
        # 从最后一个token的logits中采样
        next_token = sample(logits[-1])
        
        if next_token == tokenizer.eos_token:
            break
        
        generated.append(next_token)
    
    return tokenizer.decode(generated)
```

这个伪代码展示了几个关键的设计决策。首先，视觉编码和重新采样在推理时只进行一次，然后结果可以被多个交叉注意力层重复使用。这是一个重要的效率优化。其次，图像位置的标记和管理至关重要——模型需要知道每个文本token相对于图像的位置，以便正确地应用图像因果性约束。

对于视频输入，每一帧都通过视觉编码器处理以获得特征图，并加上可学习的时间位置编码以考虑时间序列，然后所有帧的特征被展平并连接成单一序列。这个设计优雅地扩展了对视频的支持，同时保持了整个架构的简洁性。

## 从零样本到少样本的平滑过渡

CLIP引入了零样本学习的概念，但Flamingo更进一步——它展示了少样本学习（few-shot learning）在多模态任务中的强大力量。与GPT-3能够从少数几个文本示例中学习语言任务类似，Flamingo能够从少数几个视觉-文本示例中学习视觉任务。

这个能力不是通过特殊的元学习算法实现的，而是通过一个非常简单但深刻的机制：在上下文中放入示例。论文在16个不同的任务上进行了全面的评估，包括开放式任务如视觉问答、描述任务和封闭式任务如多选视觉问答。令人印象深刻的是，使用仅仅4个示例，Flamingo就能击败之前的零样本和少样本方法。在某些情况下，仅使用32个示例且不进行任何权重调整的Flamingo，甚至超过了在数千个任务特定标注样本上微调的专门模型。

在许多基准上，Flamingo实际上超过了在数千倍以上任务特定数据上微调的模型的性能。这展示了一个深刻的现象：当模型足够大，并且在足够多样化的数据上进行了充分的预训练时，它就能够从极少的任务特定示例中快速适应。这种"在上下文中学习"的能力，后来被发现是一个贯穿所有大规模预训练模型的关键特性。

一个具体的例子说明了这一点。在一个图像说明生成任务上，模型的提示可能是这样的：

```
<image_1> a dog playing in the park
<image_2> two cats sitting on a windowsill  
<image_3> a bird flying in the sky
<image_4> ?
Generate caption:
```

模型从这四个示例中"学到"了说明生成的任务格式和风格，然后能够为第四张图片生成一个类似风格的说明。关键的是，这个学习过程完全发生在推理时，不需要任何梯度更新或参数调整。

## 训练策略的重要性

Flamingo之所以成功，并不仅仅归功于其架构，还在于其训练策略。Flamingo模型是在从网络爬取的大规模多模态数据的精心选择的混合体上训练的，完全不使用为机器学习目的标注的数据。

这种训练方式对于赋予Flamingo模型在上下文少样本学习能力是关键的。这个选择很关键。传统的多模态模型往往在小规模的、任务特定的标注数据上进行微调。而Flamingo的预训练则在数十亿级别的数据上进行。

这个规模的差异直接转化为涌现能力。当模型在足够多样化和足够大规模的数据上训练时，它自动习得了一种"通用"的多模态理解能力，这种能力可以轻松迁移到看不见的任务。这再次印证了深度学习中的一个反复验证的规律：规模确实有用。

论文还展示了模型大小与少样本性能之间的关系。Flamingo提供了三种大小，分别基于1.4B、7B和70B的Chinchilla模型，称为Flamingo-3B、Flamingo-9B和Flamingo-80B。随着模型大小的增加，少样本学习的性能稳定地改进。这个模式与GPT-3的发现完全一致：更大的模型对少样本学习更敏感。

## 对后续工作的深远影响

Flamingo发表后的影响是广泛而深远的。最直接的继承者是BLIP-2，这个模型使用了一个Q-Former来连接一个冻结的视觉编码器和一个冻结的语言模型。随后的Gemini、GPT-4V等模型虽然细节不同，但都采用了类似的思想——使用适配层连接预训练的多模态编码器和语言模型。

更深层的影响在于范式的转变。Flamingo证明了一个重要的观点：不需要从头开始训练大规模多模态模型。相反，可以通过巧妙地利用现有的预训练模型，加上精心设计的适配层和交叉注意力机制，来构建强大的多模态系统。这降低了进入这个领域的成本，因为研究者不再需要大量的计算资源来训练一个从零开始的多模态模型。只需要在有限的计算上训练适配层，就能够获得接近最先进的性能。

开源社区也受到了Flamingo的启发。OpenFlamingo项目在GitHub上获得了广泛的关注，试图复现Flamingo的能力。虽然由于计算资源的限制，开源版本的性能不如原始论文，但它使得研究者们能够深入理解和实验Flamingo的架构。

## 局限与未来方向

尽管Flamingo的成就令人瞩目，但它也有明显的局限。首先，这个架构保持了整个视觉编码器和语言模型的冻结状态，这意味着它无法适应特定领域的视觉表示。虽然这个选择提高了训练的稳定性和效率，但也限制了模型在某些特定领域的适应能力。

其次，Perceiver Resampler虽然解决了可变长度输入的问题，但它的计算成本随着输入的复杂性而增加。在处理长视频序列时，这可能成为一个瓶颈。

第三，由于模型是在互联网爬取的数据上训练的，它继承了网络数据中的各种偏见。虽然论文提到了这个问题，但没有提出有效的解决方案。

现代的多模态模型，比如LLaMA 3.2-Vision和DeepSeek-VL，已经逐渐转向了更简单的适配方案，比如使用MLP而不是Perceiver Resampler。这可能反映了一个事实：对于许多应用场景，过度复杂的架构未必必要，更简单的设计反而能更好地扩展和适应。

但Flamingo的核心洞察——通过精心设计的交叉注意力机制来融合预训练的模型——已经成为了多模态深度学习的基础范式。它不仅解决了一个特定的技术问题，更重要的是它改变了我们对多模态学习应该如何进行的理解。从"构建新模型"转变为"智能地组合现有模型"，这是一个思维方式的革命。

在多模态学习的发展路径上，CLIP打开了视觉和语言对齐的大门，而Flamingo则展示了如何通过这扇门，构建一个真正实用的、高效的、可扩展的视觉-语言系统。它的光芒也许最终会被更新的方法所超越，但它的设计哲学——尊重预训练的知识，谨慎而优雅地连接不同的模态——这些思想会长期指导多模态AI的发展。