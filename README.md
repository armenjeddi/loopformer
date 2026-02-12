# LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation (ICLR 2026)

<a target="_blank" href="">
  <img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv">
</a>
<a target="_blank" href="https://loopformer.github.io/">
  <img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat">
</a>
<a target="_blank" href="https://huggingface.co/collections/armenjeddi/loopformer">
  <img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat">
</a>

**Authors:**  
[Ahmadreza Jeddi](https://armenjeddi.github.io/), [Marco Ciccone](https://marcociccone.github.io/), [Babak Taati](https://www.cs.toronto.edu/~taati/)
<br>

![LoopFormer](assets/loopformer.png)

---

This repository contains the official implementation of **LoopFormer**.

The codebase is a fork of **NanoGPT**, and we intentionally keep it as close as possible to the original implementation for clarity and reproducibility. Beyond the looped / elastic-depth components, the main architectural difference is using **RMSNorm** instead of **LayerNorm**.

---

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm

