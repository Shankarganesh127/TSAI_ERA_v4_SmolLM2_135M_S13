
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

Token indices sequence length is longer than the specified maximum sequence length for this model (341094 > 8192). Running this sequence through the model will result in indexing errors
Loaded 341094 tokens
1 epoch = 20 batches
➡️ Starting training from scratch
Training:  10%|▉         | 499/5000 [06:28<57:55,  1.30it/s, loss=0.9962, lr=0.000300, tok/s=27851.7, dt(ms)=588.26]  

📝 Sample (step 499): The meaning of life is  to
Training:  10%|█         | 500/5000 [06:29<1:22:40,  1.10s/it, loss=0.9962, lr=0.000300, tok/s=27851.7, dt(ms)=588.26]

💾 Saved checkpoint: checkpoints/step_500.pt
Training:  20%|█▉        | 999/5000 [12:57<51:31,  1.29it/s, loss=0.5026, lr=0.000298, tok/s=27571.7, dt(ms)=594.23]  

📝 Sample (step 999): The meaning of life is  to
Training:  20%|██        | 1000/5000 [12:58<1:13:17,  1.10s/it, loss=0.5026, lr=0.000298, tok/s=27571.7, dt(ms)=594.23]

💾 Saved checkpoint: checkpoints/step_1000.pt
Training:  30%|██▉       | 1499/5000 [19:26<45:10,  1.29it/s, loss=0.0528, lr=0.000296, tok/s=27382.4, dt(ms)=598.34]  

📝 Sample (step 1499): The meaning of life is  to
Training:  30%|███       | 1500/5000 [19:27<1:03:02,  1.08s/it, loss=0.0528, lr=0.000296, tok/s=27382.4, dt(ms)=598.34]

💾 Saved checkpoint: checkpoints/step_1500.pt
Training:  40%|███▉      | 1999/5000 [25:54<38:36,  1.30it/s, loss=0.0192, lr=0.000293, tok/s=27700.5, dt(ms)=591.47]  

📝 Sample (step 1999): The meaning of life is  to
Training:  40%|████      | 2000/5000 [25:55<53:46,  1.08s/it, loss=0.0192, lr=0.000293, tok/s=27700.5, dt(ms)=591.47]

💾 Saved checkpoint: checkpoints/step_2000.pt
Training:  50%|████▉     | 2499/5000 [32:23<32:08,  1.30it/s, loss=0.0129, lr=0.000289, tok/s=27894.7, dt(ms)=587.35]

📝 Sample (step 2499): The meaning of life is  to
Training:  50%|█████     | 2500/5000 [32:24<44:25,  1.07s/it, loss=0.0129, lr=0.000289, tok/s=27894.7, dt(ms)=587.35]

💾 Saved checkpoint: checkpoints/step_2500.pt
Training:  60%|█████▉    | 2999/5000 [38:52<25:48,  1.29it/s, loss=0.0104, lr=0.000284, tok/s=27902.0, dt(ms)=587.20]

📝 Sample (step 2999): The meaning of life is  to
Training:  60%|██████    | 3000/5000 [38:53<35:54,  1.08s/it, loss=0.0104, lr=0.000284, tok/s=27902.0, dt(ms)=587.20]

💾 Saved checkpoint: checkpoints/step_3000.pt
Training:  70%|██████▉   | 3499/5000 [45:21<19:19,  1.29it/s, loss=0.0090, lr=0.000278, tok/s=27653.7, dt(ms)=592.47]

📝 Sample (step 3499): The meaning of life is  to
Training:  70%|███████   | 3500/5000 [45:22<27:15,  1.09s/it, loss=0.0090, lr=0.000278, tok/s=27653.7, dt(ms)=592.47]

💾 Saved checkpoint: checkpoints/step_3500.pt
Training:  80%|███████▉  | 3999/5000 [51:49<12:54,  1.29it/s, loss=0.0081, lr=0.000271, tok/s=28186.1, dt(ms)=581.28]

📝 Sample (step 3999): The meaning of life is  to
Training:  80%|████████  | 4000/5000 [51:50<17:50,  1.07s/it, loss=0.0081, lr=0.000271, tok/s=28186.1, dt(ms)=581.28]

💾 Saved checkpoint: checkpoints/step_4000.pt
Training:  90%|████████▉ | 4499/5000 [58:18<06:28,  1.29it/s, loss=0.0075, lr=0.000264, tok/s=27964.3, dt(ms)=585.89]

📝 Sample (step 4499): The meaning of life is  to
Training:  90%|█████████ | 4500/5000 [58:19<09:08,  1.10s/it, loss=0.0075, lr=0.000264, tok/s=27964.3, dt(ms)=585.89]

💾 Saved checkpoint: checkpoints/step_4500.pt
Training: 100%|█████████▉| 4999/5000 [1:04:47<00:00,  1.30it/s, loss=0.0073, lr=0.000256, tok/s=27943.3, dt(ms)=586.33]

📝 Sample (step 4999): The meaning of life is  to
Training: 100%|██████████| 5000/5000 [1:04:48<00:00,  1.29it/s, loss=0.0073, lr=0.000256, tok/s=27943.3, dt(ms)=586.33]

💾 Saved checkpoint: checkpoints/step_5000.pt
🎉 Training finished!


Token indices sequence length is longer than the specified maximum sequence length for this model (341094 > 8192). Running this sequence through the model will result in indexing errors
Loaded 341094 tokens
1 epoch = 20 batches
🔄 Loading checkpoint: checkpoints/step_5000.pt
🔄 Resumed from step 5000
Training: 100%|█████████▉| 499/500 [06:29<00:00,  1.29it/s, loss=0.0070, lr=0.000290, tok/s=20629.8, dt(ms)=794.19]

📝 Sample (step 5499): The meaning of life is  to
Training: 100%|██████████| 500/500 [06:30<00:00,  1.28it/s, loss=0.0070, lr=0.000290, tok/s=20629.8, dt(ms)=794.19]

💾 Saved checkpoint: checkpoints/step_5500.pt
🎉 Training finished!


details in ERA_v4_S13.ipynb notebook logs


