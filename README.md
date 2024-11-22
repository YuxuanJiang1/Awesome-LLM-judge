# Awesome-LLM-judge Survey

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![](https://img.shields.io/badge/PaperNumber-152-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red)


## Awesome-LLM-judge: A Survey
This repo include the papers discussed in our latest survey paper on Awesome-LLM-judge.    
:book: Read the full paper here: [Paper Link]()

## Reference
If our survey is useful for your research, please kindly cite our [paper]():
```
@article{wang2024comprehensive,
  title={A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness},
  author={Wang, Fali and Zhang, Zhiwei and Zhang, Xianren and Wu, Zongyu and Mo, Tzuhao and Lu, Qiuhao and Wang, Wanjing and Li, Rui and Xu, Junjie and Tang, Xianfeng and others},
  journal={arXiv preprint arXiv:2411.03350},
  year={2024}
}
```

## Overview of Awesome-LLM-judge:

![Overview of Small Language Models](images/overview_structure.png)


<!--
## Table of Contents

- [SLM Survey](#slm-survey)
  - [Table of Contents](#table-of-contents)
  - [Overview of SLMs](#overview-of-slms)
  - [Timeline of SLMs](#timeline-of-slms)
  - [SLMs Paper List](#slms-paper-list)
    - [Existing SLMs](#existing-slms)
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
-->


### Attributes

#### Helpfulness

1. <u>Constitutional AI</u>: **"Constitutional AI: Harmlessness from AI Feedback"**. *Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt,Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma,Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec,Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann,Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan* arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08073)] [[Github](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)] 
2. <u>RLAIF</u>: **"RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"**. *Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash* ICML 2024. [[Paper](https://arxiv.org/abs/2309.00267)] 
3. <u>MT-Bench</u>:**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** *Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica.* NeurIPS 2023 . [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) [[Huggingface]](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)
4. <u>Just-Eval</u>: **The unlocking spell on base llms: Rethinking alignment via in-context learning**. *Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, Yejin Choi*. ICLR 2024. [[Paper](https://openreview.net/forum?id=wxJ0eXwwda)] 
5. <u>Starling</u>: **Starling-7b: Improving helpfulness and harmlessness with rlaif**. *Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, Karthik Ganesan, Wei-Lin Chiang, Jian Zhang, Jiantao Jiao*. COLM 2024. [[Paper](https://openreview.net/forum?id=GqDntYTTbk)][[Github]](https://github.com/efrick2002/Starling)
6. <u>AUTO-J</u>: **Generative Judge for Evaluating Alignment**. *Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, Pengfei Liu*. Arxiv 2023. [[Paper](https://arxiv.org/abs/2310.05470)][[Github]](https://github.com/GAIR-NLP/auto-j)
7. <u>OAIF</u>: **Direct language model alignment from online ai feedback**. *Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, Mathieu Blondeli*. Arxiv 2024. [[Paper](https://arxiv.org/abs/2402.04792)] 

#### Harmlessness

1. **Direct preference optimization: Your language model is secretly a reward model.** *Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.* NeurIPS, 2024. [[Paper](https://arxiv.org/abs/2305.18290)] [[Code]](https://github.com/eric-mitchell/direct-preference-optimization)
2. **Enhancing chat language models by scaling high-quality instructional conversations.** *Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou.* EMNLP 2023. [[Paper]](https://aclanthology.org/2023.emnlp-main.183/) [[Code]](https://github.com/thunlp/UltraChat)
3. **SlimOrca: An Open Dataset of GPT-4 Augmented FLAN Reasoning Traces, with Verification.** *Wing Lian, Guan Wang, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium".* Huggingface, 2023. [[Data]](https://huggingface.co/datasets/Open-Orca/SlimOrca)
4. **Stanford Alpaca: An Instruction-following LLaMA model.** *Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto.* GitHub, 2023. [[Blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Github](https://github.com/tatsu-lab/stanford_alpaca)] [[HuggingFace](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff)]
5. **OpenChat: Advancing Open-source Language Models with Mixed-Quality Data.** *Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu.* ICLR, 2024. [[Paper]](https://openreview.net/forum?id=AOJyfhWYHf) [[Code]](https://github.com/imoneoi/openchat) [[HuggingFace]](https://huggingface.co/openchat)
6. **Training language models to follow instructions with human feedback.** *Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.* NeurIPS, 2022. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)
7. <u>RLHF</u>: **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al.* 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
8. <u>MobileBERT</u>: **"MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices"**. *Zhiqing Sun et al.* ACL 2020. [[Paper](https://arxiv.org/abs/2004.02984)] [[Github](https://github.com/google-research/google-research/tree/master/mobilebert)] [[HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/mobilebert)]
9. **Language models are unsupervised multitask learners.** *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.* OpenAI Blog, 2019. [[Paper]](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)

#### Reliability

1. <u>TinyStory</u>: **"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"**. *Ronen Eldan et al.* 2023. [[Paper](https://arxiv.org/abs/2305.07759)] [[HuggingFace](https://huggingface.co/papers/2305.07759)]
2. <u>AS-ES</u>: **"AS-ES Learning: Towards Efficient CoT Learning in Small Models"**. *Nuwa Xi et al.* 2024. [[Paper](https://arxiv.org/abs/2403.01969)]
3. <u>Self-Amplify</u>: **"Self-AMPLIFY: Improving Small Language Models with Self Post Hoc Explanations"**. *Milan Bhan et al.* 2024. [[Paper](https://arxiv.org/abs/2402.12038)] 
4. **Large Language Models Can Self-Improve.** *Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han.* EMNLP 2023. [[Paper]](https://aclanthology.org/2023.emnlp-main.67/) 
5. **Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing.** *Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu.* NeurIPS 2024. [[Paper]](https://openreview.net/forum?id=tPdJ2qHkOB) [[Code]](https://github.com/YeTianJHU/AlphaLLM)

#### Relevance

1. <u>GKD</u>: **"On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"**. *Rishabh Agarwal et al.* ICLR 2024. [[Paper](https://arxiv.org/abs/2306.13649)] 
2. <u>DistilLLM</u>: **"DistiLLM: Towards Streamlined Distillation for Large Language Models"**. *Jongwoo Ko et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.03898)] [[Github](https://github.com/jongwooko/distillm)]
3. <u>Adapt-and-Distill</u>: **"Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains"**. *Yunzhi Yao et al.* ACL2021. [[Paper](https://arxiv.org/abs/2106.13474)] [[Github](https://github.com/microsoft/unilm/tree/master/adalm)] 

#### Feasibility

1. <u>SmoothQuant</u>: **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**. *Guangxuan Xiao et al.* ICML 2023. [[Paper](https://arxiv.org/abs/2211.10438)] [[Github](https://github.com/mit-han-lab/smoothquant)]
2. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang et al.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
3. <u>LLM-QAT</u>: **"LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"**. *Zechun Liu et al.* 2023. [[Paper](https://arxiv.org/abs/2305.17888)]
4. <u>PB-LLM</u>: **"PB-LLM: Partially Binarized Large Language Models"**. *Yuzhang Shang et al.* 2024. [[Paper](https://openreview.net/forum?id=BifeBRhikU)] [[Github](https://github.com/hahnyuan/PB-LLM)]
5. <u>OneBit</u>: **"OneBit: Towards Extremely Low-bit Large Language Models"**. *Yuzhuang Xu et al.* NeurIPS 2024. [[Paper](https://arxiv.org/abs/2402.11295)]
6. <u>BitNet</u>: **"BitNet: Scaling 1-bit Transformers for Large Language Models"**. *Hongyu Wang et al.* 2023. [[Paper](https://arxiv.org/abs/2310.11453)]
7. <u>BitNet b1.58</u>: **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"**. *Shuming Ma et al.* 2024. [[Paper](https://arxiv.org/abs/2402.17764)]
8. <u>SqueezeLLM</u>: **"SqueezeLLM: Dense-and-Sparse Quantization"**. *Sehoon Kim et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2306.07629)] [[Github](https://github.com/SqueezeAILab/SqueezeLLM)]
9. <u>JSQ</u>: **"Compressing Large Language Models by Joint Sparsification and Quantization"**. *Jinyang Guo et al.* PMLR 2024. [[Paper](https://proceedings.mlr.press/v235/guo24g.html)] [[Github](https://github.com/uanu2002/JSQ)]
10. <u>FrameQuant</u>: **"FrameQuant: Flexible Low-Bit Quantization for Transformers"**. *Harshavardhan Adepu et al.* 2024. [[Paper](https://arxiv.org/abs/2403.06082)] [[Github](https://github.com/vsingh-group/FrameQuant)]
11. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang et al.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
12. <u>LQER</u>: **"LQER: Low-Rank Quantization Error Reconstruction for LLMs"**. *Cheng Zhang et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.02446)] [[Github](https://github.com/ChengZhang-98/lqer)]
13. <u>I-LLM</u>: **"I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models"**. *Xing Hu et al.* 2024. [[Paper](https://arxiv.org/abs/2405.17849)] [[Github](https://anonymous.4open.science/r/I-LLM-F242/README.md)]
14. <u>PV-Tuning</u>: **"PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression"**. *Vladimir Malinovskii et al.* 2024. [[Paper](https://arxiv.org/abs/2405.14852)]
15. <u>PEQA</u>: **"Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization"**. *Jeonghoon Kim et al.* NIPS 2023. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667691)]
16. <u>QLoRA</u>: **"QLORA: efficient finetuning of quantized LLMs"**. *Tim Dettmers et al.* NIPS 2023. [[Paper](https://dl.acm.org/doi/abs/10.5555/3666122.3666563)] [[Github](https://github.com/artidoro/qlora)]

#### Overall Quality

1. <u>Ma et al.</u>: **"Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!"**. *Yubo Ma et al.* EMNLP 2023. [[Paper](https://arxiv.org/abs/2303.08559)] [[Github](https://github.com/mayubo2333/LLM-IE)]
2. <u>MoQE</u>: **"Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness"**. *Young Jin Kim et al.* 2023. [[Paper](https://arxiv.org/abs/2310.02410)]
3. <u>SLM-RAG</u>: **"Can Small Language Models With Retrieval-Augmented Generation Replace Large Language Models When Learning Computer Science?"**. *Suqing Liu et al.* ITiCSE 2024. [[Paper](https://dl.acm.org/doi/10.1145/3649217.3653554)] 


## Star History

![Star History Chart](https://api.star-history.com/svg?repos=FairyFali/SLMs-Survey&type=Date)

<!-- Insertion Template: 0. <u>Model</u>: **"Title"**. *Name et al.*  202X. [[Paper]()] [[Github]()] [[HuggingFace]()] -->

