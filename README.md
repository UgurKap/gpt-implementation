# Build a Large Language Model (From Scratch) - Personal Learning Repository

This repository contains my personal implementation and experiments while working through [Sebastian Raschka's](https://sebastianraschka.com) book ["Build a Large Language Model (From Scratch)"](https://www.manning.com/books/build-a-large-language-model-from-scratch).

## About This Repository

This is a learning-focused repository where I've implemented the concepts from the book, including:

- Building a GPT-style large language model from scratch using PyTorch
- Understanding tokenization, embeddings, and attention mechanisms
- Training and fine-tuning language models
- Instruction fine-tuning techniques
- Various experiments and exercises from the book

## Repository Structure

```
.
├── Chapters/                  # Jupyter notebooks for each chapter
│   ├── Chapter2.ipynb        # Tokenization and data preparation
│   ├── Chapter3.ipynb        # Attention mechanisms
│   ├── Chapter4.ipynb        # Implementing GPT from scratch
│   ├── Chapter5.ipynb        # Pretraining on unlabeled data
│   ├── Chapter6.ipynb        # Fine-tuning for classification
│   ├── Chapter7.ipynb        # Instruction fine-tuning
│   ├── Exercise_6_*.ipynb    # Chapter 6 exercises
│   └── bells_and_whistles.ipynb  # Additional experiments
├── data/                      # Training and test data
├── models/                    # Saved model checkpoints
└── *.py                       # Helper modules and utilities
```

## Chapters Covered

- **Chapter 2**: Working with text data - tokenization and data sampling
- **Chapter 3**: Coding attention mechanisms
- **Chapter 4**: Implementing a GPT model from scratch
- **Chapter 5**: Pretraining on unlabeled data
- **Chapter 6**: Fine-tuning for classification tasks
- **Chapter 7**: Fine-tuning for instruction following

## Original Book & Resources

- **Book**: Build a Large Language Model (From Scratch)
- **Author**: Sebastian Raschka, Ph.D.
- **Publisher**: Manning Publications
- **ISBN**: 9781633437166

**Official Resources**:
- [Official Book Repository](https://github.com/rasbt/LLMs-from-scratch)
- [Book Website](https://sebastianraschka.com/llms-from-scratch/)
- [Purchase on Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch)

## Dependencies

This project uses:
- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- tiktoken (OpenAI's tokenizer)
- Additional dependencies listed in `pyproject.toml`

To install dependencies:
```bash
uv sync
```

## Attribution & License

This repository contains code adapted from and inspired by Sebastian Raschka's book and [official repository](https://github.com/rasbt/LLMs-from-scratch).

### Code Attribution
- **Original code**: Copyright (c) Sebastian Raschka under Apache License 2.0
- **Adaptations and experiments**: My personal implementations while learning from the book

### Citation
If you find this repository useful, please cite the original book:

```bibtex
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```

This repository is shared for educational purposes. The original book and code are licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

I originally made Claude Code write this README section, and I feel like it feels a bit fake to make Claude write "Special thanks ...". So, I decided to at least edit this section.  
  
I would like to thank Sebastian Raschka for preparing such a good learning resource. Because I was mostly dealing with the production side of ML recently, and/or dealing with things I understand better (e.g., uncertainty estimation), I felt a bit left behind in what has happened in the NLP field and to be honest, also got a bit intimidated. Going through the whole implementation of GPT-2, and also doing the exercises made me understand the decoder-only models better, and I also realized that I was not as far behind as I originally thought. Most of the new developments tend to be "take GPT-2, but instead do this in that layer", which makes understanding newer developments much easier. Stripping complicated topics such as this into its barebones and making it understandable is a talent Raschka clearly possesses.
