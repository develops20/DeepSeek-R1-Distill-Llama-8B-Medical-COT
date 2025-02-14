# DeepSeek-R1-Distill-Llama-8B-Medical-COT
This is a fine-tuned version of DeepSeek-R1-Distill-Llama-8B, optimized for medical reasoning and clinical case analysis using LoRA (Low-Rank Adaptation) with Unsloth.
---
library_name: transformers
pipeline_tag: text-generation
tags:
  - medical
  - deepseek
  - llama
  - unsloth
  - peft
  - transformers
  - clinical-reasoning
metrics:
  - loss
  - accuracy
---

# DeepSeek-R1-Distill-Llama-8B-Medical-COT

## 🏥 Fine-tuned Medical Model
This is a **fine-tuned version of DeepSeek-R1-Distill-Llama-8B**, optimized for **medical reasoning and clinical case analysis** using **LoRA (Low-Rank Adaptation) with Unsloth**.

- **Base Model:** [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- **Fine-Tuning Framework:** [Unsloth](https://github.com/unslothai/unsloth)
- **Dataset:** [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Quantization:** 4-bit (bitsandbytes)
- **Task:** **Clinical reasoning, medical question-answering, diagnosis assistance**
- **Pipeline Tag:** `text-generation`
- **Metrics:** `loss`, `accuracy`
- **Library Name:** `transformers`

---

## 📖 Model Details

| Feature            | Value |
|--------------------|-------------|
| **Architecture**   | Llama-8B (Distilled) |
| **Language**      | English |
| **Training Steps** | 60 |
| **Batch Size**    | 2 (with gradient accumulation) |
| **Gradient Accumulation Steps** | 4 |
| **Precision**      | Mixed (FP16/BF16 based on GPU support) |
| **Optimizer**      | AdamW 8-bit |
| **Fine-Tuned With** | PEFT + LoRA (Unsloth) |

---

## 📊 Training Summary
**Loss Trend During Fine-Tuning:**

| Step | Training Loss |
|------|--------------|
| 10   | 1.9188 |
| 20   | 1.4615 |
| 30   | 1.4023 |
| 40   | 1.3088 |
| 50   | 1.3443 |
| 60   | 1.3140 |

---

## 🚀 How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "develops20/DeepSeek-R1-Distill-Llama-8B-Medical-COT"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Run inference
def ask_model(question):
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = "A 61-year-old woman has involuntary urine loss when coughing. What would cystometry likely reveal?"
print(ask_model(question))
---
📌 Example Outputs

Q: "A 59-year-old man presents with fever, night sweats, and a 12mm aortic valve vegetation. What is the most likely predisposing factor?"

🔹 Model's Answer: "The most likely predisposing factor for this patient’s infective endocarditis is a history of valvular heart disease or prosthetic valves, given the presence of an aortic valve vegetation. The causative organism is likely Enterococcus species, which does not grow in high salt concentrations."

🔧 Fine-Tuning Details
This model was fine-tuned using Parameter Efficient Fine-Tuning (PEFT) with LoRA in Unsloth, allowing efficient adaptation without full model training.

🏆 Why Use This Model?

✅ Fine-tuned on a structured medical reasoning dataset 🔬✅ Optimized for speed with Unsloth ⚡✅ Lower VRAM usage via 4-bit quantization 🏗️✅ Handles medical Q&A, diagnosis reasoning, and case analysis 🏥

🔧 Fine-Tuning Details

This model was fine-tuned using Parameter Efficient Fine-Tuning (PEFT) with LoRA in Unsloth, allowing efficient adaptation without full model training.

Training Arguments:

TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=5,
    max_steps=60,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16=True,  # BF16 if supported
    output_dir="outputs"
)

📜 License & Contribution

License: MIT

✅ Feel free to use, modify, and improve this model. If you use it in research or projects, consider citing this work!

Contribute & Feedback: If you have suggestions or improvements, please open an issue or pull request on Hugging Face.

🤝 Acknowledgments

This model was trained with the support of Kaggle's free GPUs and the Hugging Face Transformers ecosystem. Special thanks to the Unsloth developers for optimizing LoRA fine-tuning!

