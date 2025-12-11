# 📜 Thirukkural Expert: Fine-Tuning Sarvam-1 (SLM)
Finetuned an SLM for thirukkural


This project focuses on fine-tuning **Sarvam-1**, a highly efficient Small Language Model (SLM) developed by Sarvam AI, to act as an **Expert Tamil Scholar**. 

The model is trained to interpret **Thirukkural** couplets and provide structured, deep explanations, moral messages, and real-world examples in Tamil.

## 🎯 Project Objective
The goal was to create a lightweight, domain-specific model capable of understanding classical Tamil literature and translating it into modern, relatable contexts. 

The model was fine-tuned to follow a strict output format:
1.  **தமிழ் பொருள்** (Meaning in Tamil)
2.  **ஆழமான விளக்கம்** (Deep Explanation)
3.  **கற்றுத்தரும் நன்முறை** (Moral Message)
4.  **உண்மை வாழ்விலான உதாரணம்** (Real-world Example)

## 🛠️ Tech Stack & Methodology
* **Base Model:** `sarvamai/sarvam-1` (2B parameter model optimized for Indian languages).
* **Optimization Library:** [Unsloth](https://github.com/unslothai/unsloth) (for 2x faster training and 4-bit quantization).
* **Technique:** LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
* **Framework:** Hugging Face `transformers` & `trl` (SFTTrainer).
* **Hardware:** Trained on Kaggle NVIDIA Tesla T4 (2x GPUs available, single used for training flow).

## 📊 Dataset
* **Source:** Custom generated dataset (`thirukkural_dataset_generated.json`).
* **Size:** 2,684 examples.
* **Split:** * Training: 2,415 samples
    * Validation: 269 samples
* **Format:** Instruction-tuned format using the standard Qwen chat template.

## ⚙️ Training Configuration
| Parameter | Value |
| :--- | :--- |
| **Quantization** | 4-bit (Load in 4bit) |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 16 |
| **Batch Size** | 4 per device (Grad Accum: 2) -> Effective Batch: 16 |
| **Learning Rate** | 1e-4 |
| **Epochs** | 3 |
| **Max Sequence Length** | 2048 |
| **Optimizer** | AdamW 8-bit |

## 📈 Training Results
The training process showed consistent convergence without signs of overfitting. The validation loss dropped steadily throughout the 3 epochs.

| Step | Training Loss | Validation Loss | Status |
| :--- | :--- | :--- | :--- |
| 25 | 4.6883 | 2.0585 | Initial Learning |
| 100 | 3.1166 | 1.5629 | Rapid Improvement |
| 200 | 2.9466 | 1.4913 | Steady Convergence |
| 300 | 2.8850 | 1.4598 | Fine-tuning |
| **450 (Final)** | **2.8354** | **1.4415** | **Optimal Performance** |

**Key Insight:** The drastic drop in Validation Loss (from ~2.05 to ~1.44) indicates the model successfully adapted to the persona of a Tamil Scholar and learned the structured output format required for the task.

## 🚀 How to Run Inference
You can load this model using `unsloth` or `transformers`.

```python
from unsloth import FastLanguageModel

# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "sadie26032005/sarvam-ai-finetuning-distillation-v4-1", 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Define the System Prompt used in training
sys_prompt = """**நீங்கள் ஒரு திறமையான தமிழ் அறிஞர் (Expert Tamil Scholar).
கொடுக்கப்படும் திருக்குறள்-ஐ முழுமையாகவும் ஆழமாகவும் விளக்க வேண்டும்.
உங்கள் விளக்கம் பின்வரும் வடிவத்தில் இருக்க வேண்டும்:**

1. **தமிழ் பொருள் (Meaning in Tamil)**
2. **ஆழமான விளக்கம் (Deep Explanation)**
3. **கற்றுத்தரும் நன்முறை / Moral Message**
4. **உண்மை வாழ்விலான உதாரணம் (Real-world Example)**"""

# Inference
messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "திருக்குறள்: அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு"}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")


```

outputs = model.generate(inputs, max_new_tokens=512, use_cache=True)
print(tokenizer.batch_decode(outputs)[0])
