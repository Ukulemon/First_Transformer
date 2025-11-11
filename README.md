# First Transformer

该项目实现了一个最基础的 Transformer 编码器-解码器结构，使用纯 PyTorch 模块构建，方便理解与扩展。

## 功能特点

- 自定义的多头注意力、前馈网络以及编码器/解码器层
- 简洁的 `TransformerModel` 封装，支持前向传播与贪心解码
- Sinusoidal 位置编码与自动生成的自回归 mask
- 提供 `examples/demo.py` 演示脚本，展示模型的基本用法
- 新增 `examples/train.py`，包含字符级分词、数据集构造、训练循环与推理脚本，可处理 1000 token 以内的序列

## 使用方法

1. 安装依赖（需要 PyTorch）

   ```bash
   pip install torch
   ```

2. 运行示例脚本

   ```bash
   python examples/demo.py
   ```

   输出中会展示 logits 张量的形状以及不同层的注意力权重尺寸，帮助理解模型的结构。

3. 训练一个可进行自动补全的模型（数据集中每一行的文本长度需在 1000 token 以内）：

   ```bash
   python examples/train.py train \
       --data data/train.txt \
       --output-dir checkpoints \
       --epochs 5 \
       --batch-size 32 \
       --max-length 1024
   ```

   - **训练数据输入方式**：`--data` 指向的 UTF-8 文本文件，每一行代表一条训练样本，程序会自动构建字符级分词器并按随机前缀拆分成 `src`/`tgt`。
   - **训练参数与损失存储**：所有命令行参数会写入 `checkpoints/training_config.json`，按 `--log-every` 间隔记录的平均损失保存在 `checkpoints/loss_history.csv`。
   - **模型与分词器导出**：训练完成后会在 `checkpoints/` 下生成 `transformer.pt`（模型权重与配置）及 `tokenizer.json`（字符级词表）。

4. 使用训练好的模型进行推理，根据给定前缀生成后续文本：

   ```bash
   python examples/train.py generate \
       --checkpoint checkpoints/transformer.pt \
       --tokenizer checkpoints/tokenizer.json \
       --prompt "从前有座山" \
       --max-new-tokens 200 \
       --output outputs/story.txt
   ```

   - **切换到推理模式**：通过 `generate` 子命令加载训练时保存的 `transformer.pt` 与 `tokenizer.json`，模型会自动切换到 `eval()` 状态。
   - **推理输入方式**：使用 `--prompt` 指定前缀文本，程序会将其编码为 token 序列并执行贪心生成。
   - **推理结果存储**：标准输出会打印最终的续写内容，同时当提供 `--output` 时，结果会保存到指定路径（例如 `outputs/story.txt`）。

## 项目结构

```
first_transformer/
├── __init__.py
├── attention.py
├── config.py
├── layers.py
└── utils.py
examples/
└── demo.py
```

- `config.py`：定义 `TransformerConfig` 配置类。
- `attention.py`：实现多头注意力模块。
- `layers.py`：包含编码器层与解码器层。
- `utils.py`：提供位置编码和生成 mask 的工具函数。
- `model.py`：组合上述组件，形成完整的 Transformer 模型。
- `examples/demo.py`：简单示例展示模型的使用方式。
