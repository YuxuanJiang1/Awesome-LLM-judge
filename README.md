# SLM Survey

## A Comprehensive Survey of Small Language Models: Technology, On-Device Applications, Efficiency, Enhancements for LLMs, and Trustworthiness

> TBD - Contact Information
>
> If you find this survey useful for your research, please cite the following paper:

```
[TBD - BibTeX citation link here]
```

## Table of Contents

- [SLM Survey](#slm-survey)
  - [Table of Contents](#table-of-contents)
  - [Overview of SLMs](#overview-of-slms)
  - [Timeline of SLMs](#timeline-of-slms)
  - [SLM Ppaer List](#slm-paper-list)
    - [Generic Small Language Models](#generic-small-language-models)
    - [Foundational Concepts in Building Language Models](#foundational-concepts-in-building-language-models)
    - [Advanced enhancement methods for SLM](#advanced-enhancement-methods-for-slm)
      - [Training from scratch](#training-from-scratch)
      - [Supervised fine-tuning](#supervised-fine-tuning)
      - [Data quality in KD](#data-quality-in-kd)
      - [Distillation for SLM](#distillation-for-slm)
      - [Quantization](#quantization)
      - [LLMs for SLM](#llms-for-slm)
    - [Task-specific SLM Applications](#task-specific-slm-applications)
      - [SLM in QA](#slm-in-qa)
      - [SLM in Coding](#slm-in-coding)
      - [SLM in Recommendation](#slm-in-recommendation)
      - [SLM in Web Search](#slm-in-web-search)
      - [SLM in Mobile-device](#slm-in-mobile-device)
    - [On-device Deployment Optimization Techniques](#on-device-deployment-optimization-techniques) 
      - [Memory Efficiency Optimization](#memory-efficiency-optimization)
      - [Runtime Efficiency Optimization](#runtime-efficiency-optimization)
  
## Overview of SLMs
![Overview of Small Language Models](images/overview_of_small_language_models.PNG)

## Timeline of SLMs
![Timeline of Small Language Models](images/timeline_of_small_language_models.png)

## SLM Paper List

### Generic Small Language Models

| Model Name    | Name       | # Params | Year | Paper | Code | Model | Conference |
|---------------|------------|------------|------|-------|------|-------| -----------|
| Llama 3.2     | Llama      | 1B; 3B     | 2024 |       | [Link](https://github.com/meta-llama/llama-models)     | [Link](https://huggingface.co/meta-llama/Llama-3.2-1B)           ||
| Qwen 1        | Qwen       | 1.8B; 7B; 14B; 72B| 2023 |       | [Link](https://github.com/QwenLM/Qwen)    | [Link](https://huggingface.co/Qwen)           ||
| Qwen 1.5      | Qwen       | 0.5B; 1.8B; 4B; <br> 7B; 14B; 32B; <br> 72B            | 2024 |       | [Link](https://github.com/QwenLM/Qwen)    | [Link](https://huggingface.co/Qwen)           ||
| Qwen 2        | Qwen       | 0.5B; 1.5B; 7B; <br> 57B; 72B           | 2024 |       | [Link](https://github.com/QwenLM/Qwen)    | [Link](https://huggingface.co/Qwen)           ||
| Qwen 2.5      | Qwen       | 0.5B; 1.5B; 3B; <br> 7B; 14B; 32B; 72B           | 2024 |       | [Link](https://github.com/QwenLM/Qwen)    | [Link](https://huggingface.co/Qwen)           ||
| Gemma         | Gemma 2: Improving Open Language Models at a Practical Size      | 2B; 7B     | 2024 | [Link](https://arxiv.org/abs/2408.00118)   |      |                                               ||
| Gemma 2       | Gemma 2: Improving Open Language Models at a Practical Size      | 2B; 9B; 27B| 2024 | [Link](https://arxiv.org/abs/2408.00118)   |      |                                               ||
| H2O-Danube3   | H2O-Danube3 Technical Report | 500M; 4B   | 2024 | [Link](https://arxiv.org/abs/2407.09276v1) |      | [Link](https://huggingface.co/collections/h2oai/h2o-danube3-6687a993641452457854c609)           ||
| Fox-1         | TensorOpera Unveils Fox Foundation Model: A Pioneering Small Language Model (SLM) for Cloud and Edge | 1.6B       | 2024 | [Link](https://blog.tensoropera.ai/tensoropera-unveils-fox-foundation-model-a-pioneering-open-source-slm-leading-the-way-against-tech-giants/)      |   |   ||
| Rene          | Rene       | 1.3B       | 2024     | [Link](https://www.cartesia.ai/blog/on-device)      |      | [Link](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch#:~:text=Rene%20is%20a%201.3%20billion,of%20the%20Dolma%2D1.7%20dataset.)    ||
| MiniCPM       | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | 1.2B; 2.4B | 2024     | [Link](https://arxiv.org/abs/2404.06395) | [Link](https://github.com/OpenBMB/MiniCPM-V) | [Link](https://huggingface.co/openbmb/MiniCPM-V-2) ||
| OLMo          | OLMo: Accelerating the Science of Language Models | 1B; 7B     | 2024     | [Link](https://arxiv.org/abs/2402.00838) | [Link](https://github.com/allenai/OLMo?tab=readme-ov-file)     |            ||
| TinyLlama     | TinyLlama: An Open-Source Small Language Model| 1B         | 2024     | [Link](https://arxiv.org/abs/2401.02385)    | [Link](https://github.com/jzhang38/TinyLlama)     | [Link](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)        ||
| Phi-1         | Textbooks Are All You Need | 1.3B       | 2023     | [Link](https://arxiv.org/abs/2306.11644)      |      | [Link](https://huggingface.co/microsoft/phi-1)           ||
| Phi-1.5       | Textbooks Are All You Need II: phi-1.5 technical report | 1.3B       | 2023     | [Link](https://arxiv.org/abs/2309.05463)      |      | [Link](https://huggingface.co/microsoft/phi-1_5)           ||
| Phi-2         | Phi-2: The surprising power of small language models | 2.7B       | 2023     | [Link](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)      |   | [Link](https://huggingface.co/microsoft/phi-2)           ||
| Phi-3         | Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone | 3.8B; 7B; 14B | 2024     | [Link](https://arxiv.org/abs/2404.14219)      |      | [Link](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)          ||
| Phi-3.5       | Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone | 3.8B; 4.2B; 6.6B | 2024     | [Link](https://arxiv.org/abs/2404.14219)      |      | [Link](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)           ||
| OpenELM       | OpenELM: An Efficient Language Model Family with Open Training and Inference Framework | 270M; 450M; 1.1B; <br> 3B | 2024     | [Link](https://arxiv.org/abs/2404.14619)      |  [Link](https://github.com/CarperAI/OpenELM)    | [Link](https://huggingface.co/apple/OpenELM)           ||
| MobiLlama     | MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT  | 0.5B; 0.8B    | 2024     | [Link](https://arxiv.org/abs/2402.16840)      | [Link](https://github.com/mbzuai-oryx/MobiLlama)     | [Link](https://huggingface.co/collections/MBZUAI/mobillama-65dd4182d588c91e8230332e)         ||
| MobileLLM     | MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases | 125M; 350M    | 2024     | [Link](https://arxiv.org/abs/2402.14905)      | [Link](https://github.com/facebookresearch/MobileLLM)| [Link](https://huggingface.co/papers/2402.14905)            ||
| StableLM      | StableLM   | 3B; 7B     | 2024     | | [Link](https://github.com/Stability-AI/StableLM)            | [Link](https://huggingface.co/stabilityai/stablelm-zephyr-3b)           ||
| StableLM 2    | StableLM 2 | 1.6B       | 2024     | | [Link](https://github.com/Stability-AI/StableLM)            | [Link](https://huggingface.co/stabilityai/stablelm-2-1_6b)           ||
| Cerebras-GPT  | Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster | 111M - 13B | 2023     | [Link](https://arxiv.org/abs/2304.03208)      |      | [Link](https://huggingface.co/collections/cerebras/cerebras-gpt-66c623297a2370b8e670e0a1)           ||
| Pythia        | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | 70M - 12B  | 2023     | [Link](https://arxiv.org/abs/2304.01373)      | [Link](https://github.com/EleutherAI/pythia)     | [Link](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)           ||
| BLOOM, BLOOMZ | BLOOM: A 176B-Parameter Open-Access Multilingual Language Model | 560M; 1.1B; 1.7B; <br> 3B; 7.1B; 176B           | 2023     | [Link](https://arxiv.org/abs/2211.05100)      |      | [link](https://huggingface.co/bigscience/bloom)           ||
| Galactica     | Galactica: A Large Language Model for Science | 125M; 1.3B; 6.7B; <br> 30B; 120B           | 2022     | [Link](https://arxiv.org/abs/2211.09085)      |      |            ||
| OPT           | OPT: Open Pre-trained Transformer Language Models | 125M; 350M; 1.3B; <br> 2.7B; 5.7B           | 2022     | [Link](https://arxiv.org/abs/2205.01068)      |      |            ||
| XGLM          | Few-shot Learning with Multilingual Language Models | 1.7B; 2.9B; 7.5B   | 2021     | [Link](https://arxiv.org/abs/2112.10668)      | [Link](https://github.com/facebookresearch/fairseq/tree/main/examples/xglm)     | [Link](https://huggingface.co/facebook/xglm-564M)           | EMNLP 2022 |
| GPT-Neo       | GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow | 125M; 350M; 1.3B; <br> 2.7B           | 2021     | [Link](https://zenodo.org/records/5297715)    | [Link](https://github.com/EleutherAI/gpt-neo/tree/master)     |            ||
| Megatron-gpt2 | NVIDIA Megatron-Core | 355M; 2.5B; 8.3B   | 2024     | [Link](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)       | [Link](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file)     | [Link](https://huggingface.co/nvidia/nemo-megatron-gpt-1.3B)           ||
| Minitron      | Compact Language Models via Pruning and Knowledge Distillation | 4B; 8B; 15B        | 2024     | [Link](https://arxiv.org/abs/2407.14679)      | [Link](https://github.com/NVlabs/Minitron?tab=readme-ov-file)     | [Link](https://huggingface.co/collections/nvidia/minitron-669ac727dc9c86e6ab7f0f3e)           ||
|               |            |            |      |                                               |      |                                                               ||
| Orca          | Orca: Progressive Learning from Complex Explanation Traces of GPT-4 | 7B         | 2023 | [Link](https://arxiv.org/abs/2306.02707)      |      | [Link](https://huggingface.co/microsoft/Orca-2-13b)           ||
| Orca2         | Orca 2: Teaching Small Language Models How to Reason | 13B        | 2023 | [Link](https://arxiv.org/abs/2311.11045)      |      | [Link](https://huggingface.co/microsoft/Orca-2-13b)           ||
| Dolly-v2      | Dolly      | 3B; 7B; 12B| 2023 |                                               | [Link](https://github.com/databrickslabs/dolly#getting-started-with-response-generation) | [Link](https://huggingface.co/databricks/dolly-v2-3b)         ||
| LaMini-LM     | LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions | 61M-7B     | 2023     | [Link](https://arxiv.org/abs/2304.14402)  | [Link](https://github.com/mbzuai-nlp/LaMini-LM) | [Link](https://huggingface.co/MBZUAI/LaMini-Neo-125M)                ||
| Specialized FlanT5 | Scaling Instruction-Finetuned Language Models | 250M; 760M; 3B | 2023 | [Link](https://arxiv.org/abs/2210.11416)  | [Link](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)     | [Link](https://huggingface.co/docs/transformers/v4.26.0/en/model_doc/flan-t5)         
| FlanT5        | Scaling Instruction-Finetuned Language Models | 80M; 250M; 780M; <br> 3B | 2022   | [Link](https://arxiv.org/abs/2210.11416)  | [Link](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)     | [Link](https://huggingface.co/docs/transformers/v4.26.0/en/model_doc/flan-t5)                                                               ||
| T5            | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | 60M; 220M; 770M; <br> 3B; 11B | 2019      | [Link](https://arxiv.org/abs/1910.10683)                                              | [Link](https://github.com/google-research/text-to-text-transfer-transformer)     | [Link](https://huggingface.co/docs/transformers/en/model_doc/t5)                                                              ||

<br>

### Foundational Concepts in Building Language Models

1. <u>SqueezeLLM</u>: **"SqueezeLLM: Dense-and-Sparse Quantization"**. *Sehoon Kim et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2306.07629)] [[Github](https://github.com/SqueezeAILab/SqueezeLLM)]
2. <u>JSQ</u>: **"Compressing Large Language Models by Joint Sparsification and Quantization"**. *Jinyang Guo et al.* PMLR 2024. [[Paper](https://proceedings.mlr.press/v235/guo24g.html)] [[Github](https://github.com/uanu2002/JSQ)]
3. <u>FrameQuant</u>: **"FrameQuant: Flexible Low-Bit Quantization for Transformers"**. *Harshavardhan Adepu et al.* 2024. [[Paper](https://arxiv.org/abs/2403.06082)] [[Github](https://github.com/vsingh-group/FrameQuant)]
4. <u>OneBit</u>: **"OneBit: Towards Extremely Low-bit Large Language Models"**. *Yuzhuang Xu et al.* NeurIPS 2024. [[Paper](https://arxiv.org/abs/2402.11295)]
5. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang et al.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
6. <u>LQER</u>: **"LQER: Low-Rank Quantization Error Reconstruction for LLMs"**. *Cheng Zhang et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.02446)] [[Github](https://github.com/ChengZhang-98/lqer)]
7. <u>I-LLM</u>: **"I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models"**. *Xing Hu et al.* 2024. [[Paper](https://arxiv.org/abs/2405.17849)] [[Github](https://anonymous.4open.science/r/I-LLM-F242/README.md)]
8. <u>PV-Tuning</u>: **"PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression"**. *Vladimir Malinovskii et al.* 2024. [[Paper](https://arxiv.org/abs/2405.14852)]
9. <u>BitNet</u>: **"BitNet: Scaling 1-bit Transformers for Large Language Models"**. *Hongyu Wang et al.* 2023. [[Paper](https://arxiv.org/abs/2310.11453)]
10. <u>BitNet b1.58</u>: **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"**. *Shuming Ma et al.* 2024. [[Paper](https://arxiv.org/abs/2402.17764)]
11. <u>PEQA</u>: **"Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization"**. *Jeonghoon Kim et al.* NIPS 2023. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667691)]
12. <u>QLoRA</u>: **"QLORA: efficient finetuning of quantized LLMs"**. *Tim Dettmers et al.* NIPS 2023. [[Paper](https://dl.acm.org/doi/abs/10.5555/3666122.3666563)] [[Github](https://github.com/artidoro/qlora)]
  

### Advanced enhancement methods for SLM

#### Training from scratch

1. <u>MindLLM</u>: **"MindLLM: Pre-training Lightweight Large Language Model from Scratch, Evaluations and Domain Applications"**. *Yizhe Yang et al.* 2023. [[Paper](https://arxiv.org/abs/2310.15777)] [[HuggingFace](https://huggingface.co/bit-dny/MindLLM-1b3-chat-zh-v2.0)]
2. <u>MobiLlama</u>: **"MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT"**. *Omkar Thawakar et al.* 2024. [[Paper](https://arxiv.org/abs/2402.16840)] [[Github](https://github.com/mbzuai-oryx/MobiLlama)] [[HuggingFace](https://huggingface.co/collections/MBZUAI/mobillama-65dd4182d588c91e8230332e)]
3. <u>MobileLLM</u>: **"MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"**. *Zechun Liu et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.14905)] [[Github](https://github.com/facebookresearch/MobileLLM)] [[HuggingFace](https://huggingface.co/papers/2402.14905)]


#### Supervised fine-tuning

1. <u>MobileBERT</u>: **"MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices"**. *Zhiqing Sun et al.* ACL 2020. [[Paper](https://arxiv.org/abs/2004.02984)] [[Github](https://github.com/google-research/google-research/tree/master/mobilebert)] [[HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/mobilebert)]
2. <u>Alpaca 7B</u>: **"Alpaca: A Strong, Replicable Instruction-Following Model"**. *Rohan Taori et al.* 2023. [[Paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Github](https://github.com/tatsu-lab/stanford_alpaca)] [[HuggingFace](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff)]
3. <u>RLHF</u>: **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al.* 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
4. <u>DPO</u>: **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"**. *Rafael Rafailov et al.* 2024. [[Paper](https://arxiv.org/abs/2305.18290)]

#### Data quality in KD

1. <u>TinyStory</u>: **"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"**. *Ronen Eldan et al.* 2023. [[Paper](https://arxiv.org/abs/2305.07759)] [[HuggingFace](https://huggingface.co/papers/2305.07759)]
2. <u>AS-ES</u>: **"AS-ES Learning: Towards Efficient CoT Learning in Small Models"**. *Nuwa Xi et al.* 2024. [[Paper](https://arxiv.org/abs/2403.01969)]
3. <u>Self-Amplify</u>: **"Self-AMPLIFY: Improving Small Language Models with Self Post Hoc Explanations"**. *Milan Bhan et al.* 2024. [[Paper](https://arxiv.org/abs/2402.12038)] 

#### Distillation for SLM

1. <u>GKD</u>: **"On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"**. *Rishabh Agarwal et al.* ICLR 2024. [[Paper](https://arxiv.org/abs/2306.13649)] 
2. <u>DistilLLM</u>: **"DistiLLM: Towards Streamlined Distillation for Large Language Models"**. *Jongwoo Ko et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.03898)] [[Github](https://github.com/jongwooko/distillm)]
3. <u>Adapt-and-Distill</u>: **"Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains"**. *Yunzhi Yao et al.* ACL2021. [[Paper](https://arxiv.org/abs/2106.13474)] [[Github](https://github.com/microsoft/unilm/tree/master/adalm)] 

#### Quantization

1. <u>SmoothQuant</u>: **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**. *Guangxuan Xiao et al.* ICML 2023. [[Paper](https://arxiv.org/abs/2211.10438)] [[Github](https://github.com/mit-han-lab/smoothquant)]
2. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang et al.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
3. <u>LLM-QAT</u>: **"LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"**. *Zechun Liu et al.* 2023. [[Paper](https://arxiv.org/abs/2305.17888)]
4. <u>PB-LLM</u>: **"PB-LLM: Partially Binarized Large Language Models"**. *Yuzhang Shang et al.* 202X. [[Paper](https://arxiv.org/abs/2310.00034)] [[Github](https://github.com/hahnyuan/PB-LLM)]
5. <u>OneBit</u>: **"OneBit: Towards Extremely Low-bit Large Language Models"**. *Yuzhuang Xu et al.* NeurIPS 2024. [[Paper](https://arxiv.org/abs/2402.11295)]
6. <u>BitNet</u>: **"BitNet: Scaling 1-bit Transformers for Large Language Models"**. *Hongyu Wang et al.* 2023. [[Paper](https://arxiv.org/abs/2310.11453)]
7. <u>BitNet b1.58</u>: **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"**. *Shuming Ma et al.* 2024. [[Paper](https://arxiv.org/abs/2402.17764)]

#### LLMs for SLM

1. <u>Ma et al.</u>: **"Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!"**. *Yubo Ma et al.* EMNLP 2023. [[Paper](https://arxiv.org/abs/2303.08559)] [[Github](https://github.com/mayubo2333/LLM-IE)]
2. <u>MoQE</u>: **"Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness"**. *Young Jin Kim et al.* 2023. [[Paper](https://arxiv.org/abs/2310.02410)]
3. <u>SLM-RAG</u>: **"Can Small Language Models With Retrieval-Augmented Generation Replace Large Language Models When Learning Computer Science?"**. *Suqing Liu et al.* ITiCSE 2024. [[Paper](https://dl.acm.org/doi/10.1145/3649217.3653554)] 

### Task-specific SLM Applications
#### SLM in QA

1. <u>Alpaca</u>: **"Alpaca: A Strong, Replicable Instruction-Following Model"**. *Rohan Taori et al.*  202X. [[Paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Github](https://github.com/tatsu-lab/stanford_alpaca)] [[HuggingFace](https://huggingface.co/stabilityai/StableBeluga2)] 
2. <u>Stable Beluga 7B</u>: **"Stable Beluga 2"**. *Mahan et al.*  2023. [[HuggingFace](https://huggingface.co/stabilityai/StableBeluga2)] 
3. <u>Fine-tuned BioGPT Guo et al.</u>: **"Improving Small Language Models on PubMedQA via Generative Data Augmentation"**. *Zhen Guo et al.*  2023. [[Paper](https://arxiv.org/abs/2305.07804)]
4. <u>Financial SLMs</u>: **"Fine-tuning Smaller Language Models for Question Answering over Financial Documents"**. *Karmvir Singh Phogat et al.*  2024. [[Paper](https://arxiv.org/abs/2408.12337)]
5. <u>ColBERT</u>: **"ColBERT Retrieval and Ensemble Response Scoring for Language Model Question Answering"**. *Alex Gichamba et al.*  IEEE 2024. [[Paper](https://arxiv.org/abs/2408.10808)] 
6. <u>T-SAS</u>: **"Test-Time Self-Adaptive Small Language Models for Question Answering"**. *Soyeong Jeong et al.* ACL 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.1033)] [[Github](https://github.com/starsuzi/T-SAS)]
7. <u>Rationale Ranking</u>: **"Answering Unseen Questions With Smaller Language Models Using Rationale Generation and Dense Retrieval"**. *Tim Hartill et al.*  2023. [[Paper](https://arxiv.org/abs/2308.04711)]

#### SLM in Coding

1. <u>Phi-3.5-mini</u>: **"Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"**. *Marah Abdin et al.*  202X. [[Paper](https://arxiv.org/abs/2404.14219)] [[HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)] 
2. <u>TinyLlama</u>: **"TinyLlama: An Open-Source Small Language Model"**. *Peiyuan Zhang et al.*  2024. [[Paper](https://arxiv.org/abs/2401.02385)] [[HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)] 
3. <u>CodeLlama</u>: **"Code Llama: Open Foundation Models for Code"**. *Baptiste Rozière et al.*  2024. [[Paper](https://arxiv.org/abs/2308.12950)] [[HuggingFace](https://huggingface.co/codellama)] 
4. <u>CodeGemma</u>: **"CodeGemma: Open Code Models Based on Gemma"**. *Heri Zhao et al.*  2024. [[Paper](https://arxiv.org/abs/2406.11409)] [[HuggingFace](https://huggingface.co/google/codegemma-7b-it)] 

#### SLM in Recommendation

1. <u>PromptRec</u>: **"Could Small Language Models Serve as Recommenders? Towards Data-centric Cold-start Recommendations"**. *Xuansheng Wu, et al.*  2024. [[Paper](https://arxiv.org/abs/2306.17256v4)] [[Github](https://github.com/JacksonWuxs/PromptRec)] 
2. <u>SLIM</u>: **"Can Small Language Models be Good Reasoners for Sequential Recommendation?"**. *Yuling Wang et al.*  2024. [[Paper](https://arxiv.org/abs/2403.04260v1)]
3. <u>BiLLP</u>: **"Large Language Models are Learnable Planners for Long-Term Recommendation"**. *Wentao Shi et al.*  2024. [[Paper](https://arxiv.org/abs/2403.00843v2)] 
4. <u>ONCE</u>: **"ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models"**. *Qijiong Liu et al.*  2023. [[Paper](https://arxiv.org/abs/2305.06566)] [[Github]()] [[HuggingFace]()] 
5. <u>RecLoRA</u>: **"Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation"**. *Jiachen Zhu et al.*  2024. [[Paper](https://arxiv.org/abs/2408.03533v2)] 

#### SLM in Web Search

1. <u>Content encoder</u>: **"Pre-training Tasks for Embedding-based Large-scale Retrieval"**. *Wei-Cheng Chang et al.*  ICLR 2020. [[Paper](https://arxiv.org/abs/2002.03932)]
2. <u>Poly-encoders</u>: **"Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring"**. *Samuel Humeau et al.*  ICLR 2020. [[Paper](https://arxiv.org/abs/1905.01969)]
3. <u>Twin-BERT</u>: **"TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval"**. *Wenhao Lu et al.*  2020. [[Paper](https://arxiv.org/abs/2002.06275)]
4. <u>H-ERNIE</u>: **"H-ERNIE: A Multi-Granularity Pre-Trained Language Model for Web Search"**. *Xiaokai Chu et al.*  SIGIR 2022. [[Paper](https://dl.acm.org/doi/10.1145/3477495.3531986)]
5. <u>Ranker</u>: **"Passage Re-ranking with BERT"**. *Rodrigo Nogueira et al.*  2019. [[Paper](https://arxiv.org/abs/1901.04085)] [[Github](https://github.com/nyu-dl/dl4marco-bert)]
6. <u>Rewriter</u>: **"Query Rewriting for Retrieval-Augmented Large Language Models"**. *Xinbei Ma et al.*  EMNLP2023. [[Paper](https://arxiv.org/abs/2305.14283)] [[Github](https://github.com/xbmxb/RAG-query-rewriting)]

#### SLM in Mobile-device

1. <u>Octopus</u>: **"Octopus: On-device language model for function calling of software APIs"**. *Wei Chen et al.*  2024. [[Paper](https://arxiv.org/abs/2404.01549)]
2. <u>MobileAgent</u>: **"Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration"**. *Junyang Wang et al.*  2024. [[Paper](https://arxiv.org/abs/2406.01014)] [[Github](https://github.com/X-PLUG/MobileAgent)] [[HuggingFace](https://huggingface.co/spaces/junyangwang0410/Mobile-Agent)]
3. <u>Revolutionizing Mobile Interaction</u>: **"Revolutionizing Mobile Interaction: Enabling a 3 Billion Parameter GPT LLM on Mobile"**. *Samuel Carreira et al.*  2023. [[Paper](https://arxiv.org/abs/2310.01434)] 
4. <u>AutoDroid</u>: **"AutoDroid: LLM-powered Task Automation in Android"**. *Hao Wen et al.*  2023. [[Paper](https://arxiv.org/abs/2308.15272)]
5. <u>On-device Agent for Text Rewriting</u>: **"Towards an On-device Agent for Text Rewriting"**. *Yun Zhu et al.*  2023. [[Paper](https://arxiv.org/abs/2308.11807)]

### On-device Deployment Optimization Techniques
#### Memory Efficiency Optimization

1. <u>EDGE-LLM</u>: **"EDGE-LLM: Enabling Efficient Large Language Model Adaptation on Edge Devices via Layerwise Unified Compression and Adaptive Layer Tuning and Voting"**. *Zhongzhi Yu et al.*  2024. [[Paper](https://arxiv.org/abs/2406.15758)] [[Github](https://github.com/GATECH-EIC/Edge-LLM)]
2. <u>LLM-PQ</u>: **"LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization"**. *Juntao Zhao et al.*  2024. [[Paper](https://arxiv.org/abs/2403.01136)] [[Github](https://github.com/tonyzhao-jt/LLM-PQ?tab=readme-ov-file)]
3. <u>AWQ</u>: **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"**. *Ji Lin et al.*  MLSys 2024. [[Paper](https://arxiv.org/abs/2306.00978)] [[Github](https://github.com/mit-han-lab/llm-awq)]
4. <u>MobileAIBench</u>: **"MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases"**. *Rithesh Murthy et al.*  2024. [[Paper](https://arxiv.org/abs/2406.10290)] [[Github](https://github.com/XiaoMi/mobile-ai-bench)]
5. <u>MobileLLM</u>: **"MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"**. *Zechun Liu et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.14905)] [[Github](https://github.com/facebookresearch/MobileLLM)] [[HuggingFace](https://huggingface.co/papers/2402.14905)]
6. <u>EdgeMoE</u>: **"EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models"**. *Rongjie Yi et al.*  2023. [[Paper](https://arxiv.org/abs/2308.14352)] [[Github](https://github.com/sharc-lab/Edge-MoE)] 
7. <u>GEAR</u>: **"GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM"**. *Hao Kang et al.*  2024. [[Paper](https://arxiv.org/abs/2403.05527)] [[Github](https://github.com/opengear-project/GEAR)]
8. <u>DMC</u>: **"Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference"**. *Name et al.*  202X. [[Paper](https://arxiv.org/abs/2403.09636)]
9. <u>Transformer-Lite</u>: **"Transformer-Lite: High-efficiency Deployment of Large Language Models on Mobile Phone GPUs"**. *Luchang Li et al.*  202X. [[Paper](https://arxiv.org/abs/2403.20041)]
10. <u>LLMaaS</u>: **"Wangsong Yin"**. *Piotr Nawrot et al.*  2024. [[Paper](https://arxiv.org/abs/2403.11805)]

#### Runtime Efficiency Optimization

1. <u>EdgeMoE</u>: **"EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models"**. *Rongjie Yi et al.*  2023. [[Paper](https://arxiv.org/abs/2308.14352)] [[Github](https://github.com/sharc-lab/Edge-MoE)] 
2. <u>LLMCad</u>: **"LLMCad: Fast and Scalable On-device Large Language Model Inference"**. *Daliang Xu et al.*  2023. [[Paper](https://arxiv.org/abs/2309.04255)]
3. <u>LinguaLinked</u>: **"LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices"**. *Junchen Zhao et al.*  2023 [[Paper](https://arxiv.org/abs/2312.00388)]



<!-- Insertion Template: 0. <u>Model</u>: **"Title"**. *Name et al.*  202X. [[Paper]()] [[Github]()] [[HuggingFace]()] -->
