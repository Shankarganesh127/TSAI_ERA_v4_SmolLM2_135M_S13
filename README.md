
# 🧠 SmolLM-135M — Custom LLaMA-Style Model (Trained From Scratch)

This repository contains **SmolLM-135M**, a fully custom **decoder-only Transformer** inspired by LLaMA architecture.
The model is implemented **entirely from scratch in PyTorch**, including:

* LLaMA-style RMSNorm
* Rotary Positional Embeddings (RoPE)
* SDPA / FlashAttention-class fused attention
* Grouped Query Attention (GQA)
* SiLU-gated Feed-Forward Network
* Weight-tied LM head
* bfloat16 + TF32 training for speed

This makes the model **small, fast, and trainable on a single GPU**.

---

# 📐 Architecture Overview

SmolLM-135M is a compact GPT-like model built using **30 decoder layers** and:

| Component           | Value                                  |
| ------------------- | -------------------------------------- |
| Model Type          | Decoder-only Transformer (LLaMA-style) |
| Hidden Size         | **576**                                |
| FFN Size            | **1536**                               |
| Attention Heads     | **9**                                  |
| KV Heads (GQA)      | **3**                                  |
| Layers              | **30**                                 |
| Sequence Length     | **8192**                               |
| Vocabulary          | **49152 tokens**                       |
| Positional Encoding | **RoPE (θ = 100000)**                  |
| Activation          | **SiLU**                               |
| LayerNorm           | **RMSNorm (eps=1e-5)**                 |
| Attention           | **SDPA fused attention**               |
| Precision           | **bfloat16 + TF32**                    |
| Weight Tying        | Enabled                                |

The structure of each transformer block:

```
Input
 └─ RMSNorm
     └─ Multi-Head Attention (SDPA + RoPE + GQA)
         └─ Residual
             └─ RMSNorm
                 └─ Feed Forward Network (SiLU gate)
                     └─ Residual → Output
```

---

# 🧬 Configuration

Below is the exact model configuration used:

```json
{
  "model_type": "llama",
  "vocab_size": 49152,
  "hidden_size": 576,
  "intermediate_size": 1536,
  "num_hidden_layers": 30,
  "num_attention_heads": 9,
  "num_key_value_heads": 3,
  "max_position_embeddings": 8192,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-5,
  "initializer_range": 0.041666666666666664,
  "rope_theta": 100000,
  "rope_interleaved": false,
  "attention_dropout": 0.0,
  "attention_bias": false,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "pretraining_tp": 1,
  "is_llama_config": true,
  "transformers_version": "4.40.1"
}
```

---

# ⚡ Training Details

Training is performed using:

* **bfloat16 mixed precision** (`torch.autocast`)
* **SDPA fused attention kernels**
* **TF32 matmul acceleration**
* **Cosine Annealing LR scheduler**
* **Gradient accumulation**
* **Batch size 16 × sequence length 1024**
* **Custom minimal dataloader for efficiency**

Checkpointing includes:

* model weights
* optimizer state
* LR scheduler
* CPU + CUDA RNG states

allowing **exact reproducibility**.

---

# 🔮 Sample Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Shankar7ganesh/SmolLM-135M")
tokenizer = AutoTokenizer.from_pretrained("Shankar7ganesh/SmolLM-135M")

prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0]))
```

---

# 🖥️ Inference (Gradio Space)

This Space provides an interactive UI where you can:

* Enter a text prompt
* Adjust generation settings (temperature, max tokens, top-p, etc.)
* Generate text using SmolLM-135M

---

# 📝 Notes

* This is a **research-grade minimal implementation** of a LLaMA-style architecture.
* The model is intentionally small (135M parameters) to ensure reproducibility and fast experimentation.
* Training can be extended using custom datasets, instruction tuning, or RLHF.

---

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttentionSDPA(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
          (rope): RotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act): SiLU()
        )
        (input_layernorm): RMSNorm()
        (post_attention_layernorm): RMSNorm()
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)

<img width="283" height="331" alt="image" src="https://github.com/user-attachments/assets/9b13aa87-8163-4e08-ae46-3b3c61669fcc" />
