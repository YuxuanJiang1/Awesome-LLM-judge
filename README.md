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

### Foundational Concepts in Building Language Models

| Method        | Bit           | Type  | Notable Benefits                       | Year      | Paper       | Code   | Model Card |
|---------------| --------------|-------|----------------------------------------|-----------|-------------|----------|--------------|
| SqueezeLLM    | 3-bit         | PTQ   | Ultra-low bit quantization             | 2023      | [Link](https://arxiv.org/abs/2306.07629)               | [Link](https://github.com/SqueezeAILab/SqueezeLLM)               |   | 
| JSQ           | Flexible      | PTQ   | Better compression-accuracy trade-offs | 2024      | [Link](https://proceedings.mlr.press/v235/guo24g.html) | [Link](https://github.com/uanu2002/JSQ)                          |   |
| FrameQuant    | Fractional bit| PTQ   | Better compression-accuracy trade-offs | 2024      | [Link](https://arxiv.org/abs/2403.06082)               | [Link](https://github.com/vsingh-group/FrameQuant)               |   |
| OneBit        | 1-bit         | PTQ   | 1-bit quantization                     | 2024      | [Link](https://arxiv.org/abs/2402.11295)               |                                                                  |   |
| BiLLM         | 1-bit         | PTQ   | 1-bit quantization                     | 2024      | [Link](https://arxiv.org/abs/2402.04291)               | [Link](https://github.com/Aaronhuang-778/BiLLM)                  |   |
| LQER          | Flexible      | PTQ   | Better compression-accuracy trade-offs | 2024      | [Link](https://arxiv.org/abs/2402.02446)               | [Link](https://github.com/ChengZhang-98/lqer)                    |   |
| I-LLM         | Flexible      | PTQ   | Integer-only Quantization              | 2024      | [Link](https://arxiv.org/abs/2405.17849)               | [Link](https://anonymous.4open.science/r/I-LLM-F242/README.md)   |   |
| PV-Tuning     | 1-bit/2-bit   | PTQ   | Better compression-accuracy trade-offs | 2024      | [Link](https://arxiv.org/abs/2405.14852)               |                                                                  |   |
| BitNet        | 1-bit         | QAT   | 1-bit quantization                     | 2023      | [Link](https://arxiv.org/abs/2310.11453)               |                                                                  |   |
| BitNet b1.58  | {-1, 0, 1}    | QAT   | 1-bit quantization                     | 2024      | [Link](https://arxiv.org/abs/2402.17764)               |                                                                  |   |
| PEQA          | Flexible      | QAT   | Parameter-Efficient Finetuning         | 2024      | [Link](https://dl.acm.org/doi/10.5555/3666122.3667691) |                                                                  |   |
| QLoRA         | NF4           | QAT   | Parameter-Efficient Finetuning         | 2024      | [Link](https://dl.acm.org/doi/abs/10.5555/3666122.3666563) | [Link](https://github.com/artidoro/qlora)                    | [Link](https://huggingface.co/timdettmers)          

### Advanced enhancement methods for SLM

#### Training from scratch

| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| MindLLM       | 2023     | [Link](https://arxiv.org/abs/2310.15777)      |      | [Link](https://huggingface.co/bit-dny/MindLLM-1b3-chat-zh-v2.0)           |
| MobiLlama     | 2024     | [Link](https://arxiv.org/abs/2402.16840)      | [Link](https://github.com/mbzuai-oryx/MobiLlama)     | [Link](https://huggingface.co/collections/MBZUAI/mobillama-65dd4182d588c91e8230332e)         |
| MobileLLM     | 2024     | [Link](https://arxiv.org/abs/2402.14905)      | [Link](https://github.com/facebookresearch/MobileLLM)| [Link](https://huggingface.co/papers/2402.14905)            |

#### Supervised fine-tuning
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| MobileBERT    | 2020     | [Link](https://arxiv.org/abs/2004.02984)      | [Link](https://github.com/google-research/google-research/tree/master/mobilebert)     | [Link](https://huggingface.co/docs/transformers/en/model_doc/mobilebert)           |
| Alpaca 7B     | 2023     | [Link](https://crfm.stanford.edu/2023/03/13/alpaca.html)      | [Link](https://github.com/tatsu-lab/stanford_alpaca)    | [Link](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff)           |
| RLHF          | 2022     | [Link](https://arxiv.org/abs/2203.02155)      |      |            |
| DPO           | 2024     | [Link](https://arxiv.org/abs/2305.18290)      |      |            |

#### Data quality in KD
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| TinyStory     | 2023     | [Link](https://arxiv.org/abs/2305.07759)      |      | [Link](https://huggingface.co/papers/2305.07759)           |
| AS-ES         | 2024     | [Link](https://arxiv.org/abs/2403.01969)      |      |            |
| Self-Amplify  | 2024     | [Link](https://arxiv.org/abs/2402.12038)      |      |            |

#### Distillation for SLM
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| GKD           | 2024     | [Link](https://arxiv.org/abs/2306.13649)      |      |            |
| DistilLLM     | 2024     | [Link](https://arxiv.org/abs/2402.03898)      | [Link](https://github.com/jongwooko/distillm)     |            |
| Adapt-and-Distill | 2021 | [Link](https://arxiv.org/abs/2106.13474)      | [Link](https://github.com/microsoft/unilm/tree/master/adalm)     |            |

#### Quantization
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| SmoothQuant   | 2023     | [Link](https://arxiv.org/abs/2211.10438)      | [Link](https://github.com/mit-han-lab/smoothquant)    |            |
| BiLLM         | 2024     | [Link](https://arxiv.org/abs/2402.04291)      | [Link](https://github.com/Aaronhuang-778/BiLLM)     |            |
| LLM-QAT       | 2023     | [Link](https://arxiv.org/abs/2305.17888)      |      |            |
| PB-LLM        | 2023     | [Link](https://arxiv.org/abs/2310.00034)      | [Link](https://github.com/hahnyuan/PB-LLM)    |            |
| OneBit        | 2024     | [Link](https://arxiv.org/abs/2402.11295)     |      |            |
| BitNet        | 2023     | [Link](https://arxiv.org/abs/2310.11453)       |      |            |
| BitNet b1.58  | 2024     | [Link](https://arxiv.org/abs/2402.17764)  |      |            |

#### LLMs for SLM
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| Ma et al.     | 2023     | [Link](https://arxiv.org/abs/2303.08559)      | [Link](https://github.com/mayubo2333/LLM-IE)     |            |
| MoQE          | 2023     | [Link](https://arxiv.org/abs/2310.02410)      |      |            |
| SLM-RAG       | 2024     | [Link](https://dl.acm.org/doi/10.1145/3649217.3653554)      |      |            |


### Task-specific SLM Applications
#### SLM in QA
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| Alpaca        | 2023     | [Link](https://crfm.stanford.edu/2023/03/13/alpaca.html)      | [Link](https://github.com/tatsu-lab/stanford_alpaca)    | [Link](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff)  |
| Stable Beluga 7B  |      |   |      | [Link](https://huggingface.co/stabilityai/StableBeluga2)           |
| Fine-tuned BioGPT Guo et al.  | 2023     | [Link](https://arxiv.org/abs/2305.07804)          |      |            |
| Financial SLMs | 2024     | [Link](https://arxiv.org/abs/2408.12337)      |      |            |
| ColBERT        | 2024     | [Link](https://arxiv.org/abs/2408.10808)      |      |            |
| T-SAS          | 2023     | [Link](https://aclanthology.org/2023.findings-emnlp.1033/)      | [Link](https://github.com/starsuzi/T-SAS)     |            |
| Rationale Ranking     | 2023      | [Link](https://arxiv.org/abs/2308.04711)     |      |            |

#### SLM in Coding
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| Phi-3.5-mini    | 2024     | [Link](https://arxiv.org/abs/2404.14219)      |      | [Link](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)           |
| TinyLlama       | 2024     | [Link](https://arxiv.org/abs/2401.02385)    | [Link](https://github.com/jzhang38/TinyLlama)     | [Link](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)        |
| CodeLlama       | 2024     | [Link](https://arxiv.org/abs/2308.12950)       |            | [Link](https://huggingface.co/codellama)
| CodeGemma       | 2024     | [Link](https://arxiv.org/abs/2406.11409)      |      | [Link](https://huggingface.co/google/codegemma-7b-it)           |

#### SLM in Recommendation
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| PromptRec     | 2024     | [Link](https://arxiv.org/abs/2306.17256v4)      | [Link](https://github.com/JacksonWuxs/PromptRec)     |            |
| SLIM          | 2024     | [Link](https://arxiv.org/abs/2403.04260v1)      |      |            |
| BiLLP         | 2024     | [Link](https://arxiv.org/abs/2403.00843v2)      | [Link](https://github.com/jizhi-zhang/BiLLP)     |            |
| LLaMa-2-7B as an Encoder | 2024     | [Link](https://arxiv.org/abs/2305.06566)      |           |  |
| RecLoRA       | 2024     | [Link](https://arxiv.org/abs/2408.03533v2)      |      |            |

#### SLM in Web Search
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| Content encoder     | 2020     | [Link](https://arxiv.org/abs/2002.03932)      |      |            |
| Content encoder (Poly-encoders)     | 2020     | [Link](https://arxiv.org/abs/1905.01969)      |      |            |
| Content encoder (Twin-BERT)     | 2020     | [Link](https://arxiv.org/abs/2002.06275)      |      |            |
| Ranker (H-ERNIE)          | 2022     | [Link](https://dl.acm.org/doi/10.1145/3477495.3531986)      |      |            |
| Ranker          | 2019     | [Link](https://arxiv.org/abs/1901.04085)      | [Link](https://github.com/nyu-dl/dl4marco-bert)     |            |
| Rewriter       | 2023     | [Link](https://arxiv.org/abs/2305.14283)      |      |            |

#### SLM in Mobile-device
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| Octopus     | 2024     | [Link](https://arxiv.org/abs/2404.01549)      |      |            |
| MobileAgent | 2024     | [Link](https://arxiv.org/abs/2406.01014)      | [Link](https://github.com/X-PLUG/MobileAgent)     | [Link](https://huggingface.co/spaces/junyangwang0410/Mobile-Agent)           |
| Revolutionizing Mobile Interaction       | 2023     | [Link](https://arxiv.org/abs/2310.01434)      |      |            |
| AutoDroid       | 2023     | [Link](https://arxiv.org/abs/2308.15272)      | [Link](https://github.com/MobileLLM/AutoDroid)     |            |
| On-device Agent for Text Rewriting       | 2023     | [Link](https://arxiv.org/abs/2308.11807)      |      |            |


### On-device Deployment Optimization Techniques
#### Memory Efficiency Optimization
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| EDGE-LLM     | 2024     | [Link](https://arxiv.org/abs/2406.15758)      | [Link](https://github.com/GATECH-EIC/Edge-LLM)     |            |
| LLM-PQ          | 2024      | [Link](https://arxiv.org/abs/2403.01136)       | [Link](https://github.com/tonyzhao-jt/LLM-PQ?tab=readme-ov-file)      |            |
| AWQ       | 2024     |       | [Link](https://arxiv.org/abs/2306.00978)     | [Link](https://github.com/mit-han-lab/llm-awq)           |
| MobileAIBench       | 2024     | [Link](https://arxiv.org/abs/2406.10290)      | [Link](https://github.com/XiaoMi/mobile-ai-bench)      |            |
| MobileLLM     | 2024     | [Link](https://arxiv.org/abs/2402.14905)      | [Link](https://github.com/facebookresearch/MobileLLM)| [Link](https://huggingface.co/papers/2402.14905)            |
| EdgeMoE       | 2023     | [Link](https://arxiv.org/abs/2308.14352)      | [Link](https://github.com/sharc-lab/Edge-MoE)     |            |
| GEAR       | 2024     | [Link](https://arxiv.org/abs/2403.05527)      | [Link](https://github.com/opengear-project/GEAR)     |            |
| DMC       | 2024     | [Link](https://arxiv.org/abs/2403.09636)      |      |            |
| Transformer-Lite        | 2024     | [Link](https://arxiv.org/abs/2403.20041)      |      |            |
| LLMaaS        | 2024     | [Link](https://arxiv.org/abs/2403.11805)      |      |            |

#### Runtime Efficiency Optimization
| Method        | Year | Paper | Code | Model Card |
|---------------|------|-------|------|------------|
| EdgeMoE     | 2023     | [Link](https://arxiv.org/abs/2308.14352)      | [Link](https://github.com/sharc-lab/Edge-MoE)     |            |
| LLMCad          | 2023     | [Link](https://arxiv.org/abs/2309.04255)      |      |            |
| LinguaLinked       | 2023     | [Link](https://arxiv.org/abs/2312.00388)      |      |            |
