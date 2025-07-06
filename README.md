# Estimating Computational Complexity with Large Language Models

This repository contains the code, experiments, and documentation for my undergraduate thesis titled:

**"Estimating Computational Complexity with Large Language Models"**

## 🧠 Overview

The goal of this project is to investigate the potential of large language models (LLMs) to estimate the computational complexity of algorithms **without executing them**. Traditionally, this estimation requires either empirical testing with inputs of varying size or manual step-counting. Here, we explore a novel approach based on deep learning and natural language processing.

This research aims to support software developers in writing more efficient code by enabling LLMs to infer time or space complexity directly from algorithm descriptions or source code.

## 📂 Structure

```bash
.
├── data/                # Raw and processed datasets for training and evaluation
├── figures/             # Figures used in the thesis
├── notebooks/           # Jupyter notebooks for training and experimentation
├── outputs/             # Output files such as fine-tuned models and inference results
├── src/                 # Source code (Python scripts and utilities)
├── thesis/              # LaTeX source and compiled PDF of the thesis
├── .gitignore           # Git ignore rules for excluding unnecessary files from version control
└── README.md            # Project overview and instructions
```

### 🛠️ Installation

To run the fine-tuning and inference scripts, install the following dependencies:

```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
pip install --no-deps unsloth
```

Make sure you're using Python ≥3.10 and a GPU-enabled environment with CUDA support. These dependencies are optimized for 4-bit quantized models and efficient training with LoRA.

## 📊 Methodology

1. **Dataset Preparation**: Collection of algorithms annotated with their computational complexity.
2. **Prompt Engineering**: Designing prompts to elicit accurate predictions from LLMs.
3. **Fine-Tuning**: Training LLMs on algorithm-complexity pairs using QLoRA.
4. **Evaluation**: Testing generalization on unseen algorithms and measuring prediction accuracy.

## 📈 Goals

- Investigate in-context learning capabilities for reasoning about code.
- Evaluate whether LLMs can estimate complexity without execution.
- Compare zero-shot, few-shot, and fine-tuned performance.

## 📚 References

- Albert, J. V., Rabasa, F. J. F., & Quetglás, G. M. (1998). *Introducció a l’anàlisi i disseny d’algorismes*. Universitat de València.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). [Layer normalization](https://arxiv.org/abs/1607.06450). *arXiv preprint arXiv:1607.06450*.
- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). [Language models are few-shot learners](https://arxiv.org/abs/2005.14165). *Advances in Neural Information Processing Systems, 33*, 1877–1901.
- Hochreiter, S., & Schmidhuber, J. (1997). [Long short-term memory](https://doi.org/10.1162/neco.1997.9.8.1735). *Neural Computation, 9*(8), 1735–1780.
- Howard, J., & Ruder, S. (2018). [Universal language model fine-tuning for text classification](https://doi.org/10.18653/v1/P18-1031). In *Proceedings of ACL 2018* (pp. 328–339).
- Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing* (3rd ed.). Manuscript in preparation. [Link](https://web.stanford.edu/~jurafsky/slp3)
- Kingma, D. P., & Ba, J. (2014). [Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980). *arXiv preprint arXiv:1412.6980*.
- Lin, C.-Y. (2004). [ROUGE: A package for automatic evaluation of summaries](https://aclanthology.org/W04-1013/). In *Proceedings of the Workshop on Text Summarization Branches Out* (pp. 74–81).
- Loshchilov, I., & Hutter, F. (2019). [Decoupled weight decay regularization](https://arxiv.org/abs/1711.05101). *arXiv preprint arXiv:1711.05101*.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). [BLEU: A method for automatic evaluation of machine translation](https://doi.org/10.3115/1073083.1073135). In *Proceedings of ACL 2002* (pp. 311–318).
- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). [SQuAD: 100,000+ questions for machine comprehension of text](https://aclanthology.org/D16-1264/). In *Proceedings of EMNLP 2016* (pp. 2383–2392).
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0). *Nature, 323*(6088), 533–536.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). In *Advances in Neural Information Processing Systems* (Vol. 30).


## 🧑‍💻 Author

**Daniel Sánchez Pagán**  
Bachelor's Degree in Mathematics  
University of Alicante  
Academic Year: 2024–2025

## 📄 License

This project is for academic and research purposes only.  
License details will be added later.
