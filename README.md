# SLM Survey

## A Comprehensive Survey of Small Language Models: Technology, On-Device Applications, Efficiency, Enhancements for LLMs, and Trustworthiness

## Overview of SLMs
![Overview of Small Language Models](images/overview_structure.png)

## Timeline of SLMs
![Timeline of Small Language Models](images/timeline.png)

## SLMs Paper List
### Existing SLMs


| Model | #Params | Date | Paradigm | Domain | Code Link | Paper/Blog Link |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Llama 3.2 | 1B; 3B | 2024.9 | Pre-train | Generic | [HF](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) | [Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |
| Qwen 1 | 1.8B; 7B; 14B; 72B | 2023.12 | Pre-train | Generic | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2309.16609) |
| Qwen 1.5 | 0.5B; 1.8B; 4B; 7B; 14B; 32B; 72B | 2024.2 | Pre-train | Generic | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2309.16609) |
| Qwen 2 | 0.5B; 1.5B; 7B; 57B; 72B | 2024.6 | Pre-train | Generic | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2407.10671) |
| Qwen 2.5 | 0.5B; 1.5B; 3B; 7B; 14B; 32B; 72B | 2024.9 | Pre-train | Generic | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2407.10671) |
| Gemma | 2B; 7B | 2024.2 | Pre-train | Generic | [HF](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b) | [Paper](https://arxiv.org/abs/2403.08295) |
| Gemma 2 | 2B; 9B; 27B | 2024.7 | Pre-train | Generic | [HF](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) | [Paper](https://arxiv.org/abs/2408.00118) |
| H2O-Danube3 | 500M; 4B | 2024.7 | Pre-train | Generic | [HF](https://huggingface.co/collections/h2oai/h2o-danube3-6687a993641452457854c609) | [Paper](https://arxiv.org/abs/2407.09276) |
| Fox-1 | 1.6B | 2024.6 | Pre-train | Generic | [HF](https://huggingface.co/tensoropera/Fox-1-1.6B) | [Blog](https://blog.tensoropera.ai/tensoropera-unveils-fox-foundation-model-a-pioneering-open-source-slm-leading-the-way-against-tech-giants/) |
| Rene | 1.3B | 2024.5 | Pre-train | Generic | [HF](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch) | [Paper](https://cartesia.ai/blog/on-device) |
| MiniCPM | 1.2B; 2.4B | 2024.4 | Pre-train | Generic | [HF](https://huggingface.co/collections/openbmb/minicpm-65d48bf958302b9fd25b698f) | [Paper](https://arxiv.org/abs/2404.06395) |
| OLMo | 1B; 7B | 2024.2 | Pre-train | Generic | [HF](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) | [Paper](https://arxiv.org/abs/2402.00838) |
| TinyLlama | 1B | 2024.1 | Pre-train | Generic | [HF](https://huggingface.co/TinyLlama) | [Paper](https://arxiv.org/abs/2401.02385) |
| Phi-1 | 1.3B | 2023.6 | Pre-train | Coding | [HF](https://huggingface.co/microsoft/phi-1) | [Paper](https://arxiv.org/abs/2306.11644) |
| Phi-1.5 | 1.3B | 2023.9 | Pre-train | Generic | [HF](https://huggingface.co/microsoft/phi-1_5) | [Paper](https://arxiv.org/abs/2309.05463) |
| Phi-2 | 2.7B | 2023.12 | Pre-train | Generic | [HF](https://huggingface.co/microsoft/phi-2) | [Paper](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) |
| Phi-3 | 3.8B; 7B; 14B | 2024.4 | Pre-train | Generic | [HF](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | [Paper](https://arxiv.org/abs/2404.14219) |
| Phi-3.5 | 3.8B; 4.2B; 6.6B | 2024.4 | Pre-train | Generic | [HF](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | [Paper](https://arxiv.org/abs/2404.14219) |
| OpenELM | 270M; 450M; 1.1B; 3B | 2024.4 | Pre-train | Generic | [HF](https://huggingface.co/apple/OpenELM) | [Paper](https://openreview.net/forum?id=XNMbTkxroF) |
| MobiLlama | 0.5B; 0.8B | 2024.2 | Pre-train | Generic | [HF](https://huggingface.co/apple/OpenELM) | [Paper](URL) |
| MobileLLM | 125M; 350M | 2024.2 | Pre-train | Generic | [HF](https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95) | [Paper](https://arxiv.org/abs/2402.14905) |
| StableLM | 3B; 7B | 2023.4 | Pre-train | Generic | [HF](https://huggingface.co/stabilityai/stablelm-3b-4e1t) | [Paper](https://huggingface.co/stabilityai/stablelm-3b-4e1t) |
| StableLM 2 | 1.6B | 2024.2 | Pre-train | Generic | [HF](https://huggingface.co/stabilityai/stablelm-2-1_6b) | [Paper](https://arxiv.org/abs/2402.17834) |
| Cerebras-GPT | 111M-13B | 2023.4 | Pre-train | Generic | [HF](https://huggingface.co/collections/cerebras/cerebras-gpt-66c623297a2370b8e670e0a1) | [Paper](https://arxiv.org/abs/2304.03208) |
| BLOOM, BLOOMZ | 560M; 1.1B; 1.7B; 3B; 7.1B; 176B | 2022.11 | Pre-train | Generic | [HF](https://huggingface.co/bigscience) | [Paper](https://arxiv.org/abs/2211.05100) |
| OPT | 125M; 350M; 1.3B; 2.7B; 5.7B | 2022.5 | Pre-train | Generic | [HF](https://huggingface.co/facebook/opt-350m) | [Paper](https://arxiv.org/abs/2205.01068) |
| XGLM | 1.7B; 2.9B; 7.5B | 2021.12 | Pre-train | Generic | [HF](https://huggingface.co/facebook/xglm-564M) | [Paper](https://aclanthology.org/2022.emnlp-main.616) | 
| GPT-Neo | 125M; 350M; 1.3B; 2.7B | 2021.5 | Pre-train | Generic | [Github](https://github.com/EleutherAI/gpt-neo) | [Paper](https://zenodo.org/records/5297715) |
| Megatron-gpt2 | 355M; 2.5B; 8.3B | 2019.9  | Pre-train | Generic | [Github](https://github.com/NVIDIA/Megatron-LM) | [Paper](https://arxiv.org/abs/1909.08053), [Blog](https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm) |
| MINITRON | 4B; 8B; 15B | 2024.7 | Pruning and Distillation | Generic | [HF](https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base)| [Paper](https://arxiv.org/abs/2407.14679) |
| Orca 2 | 7B | 2023.11 | Distillation | Generic | [HF](https://huggingface.co/microsoft/Orca-2-7b) |[Paper](https://arxiv.org/abs/2311.11045) |
| Dolly-v2 | 3B; 7B; 12B | 2023.4 | Instruction tuning | Generic | [Code](https://huggingface.co/databricks/dolly-v1-6b) | [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| LaMini-LM | 61M-7B | 2023.4 | Distillation | Generic | [HF](https://huggingface.co/databricks/dolly-v1-6b) | [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| Specialized FlanT5 | 250M; 760M; 3B | 2023.1 | Instruction Tuning | Generic (math) | - | [Paper](https://proceedings.mlr.press/v202/fu23d.html) |
| FlanT5 | 80M; 250M; 780M; 3B | 2022.10 | Instruction Tuning | Generic | [HF](https://huggingface.co/google/flan-t5-xxl) | [Paper](https://arxiv.org/abs/2210.11416) |
| T5 | 60M; 220M; 770M; 3B; 11B | 2019.9 | Pre-train | Generic | [HF](https://huggingface.co/google/t5-v1_1-base) | [Paper](https://arxiv.org/abs/1910.10683) |
