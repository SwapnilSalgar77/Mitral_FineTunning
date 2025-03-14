# Mitral_FineTunning
A step-by-step guide to logging into Hugging Face in Google Colab, including installation, authentication, and verification. Also includes a basic guide to fine-tuning Large Language Models (LLMs) using Hugging Face's transformers library

# Mistral-7B Fine-Tuning with LoRA on Samsum Dataset

This repository contains the code and configuration for fine-tuning **Mistral-7B-Instruct** using **LoRA (Low-Rank Adaptation)** and **4-bit quantization** on the **Samsum dialogue summarization dataset**.

## 🚀 Features
- **Fine-tunes Mistral-7B-Instruct with LoRA** (efficient parameter tuning).
- **Uses 4-bit quantization (BitsAndBytes)** for low VRAM consumption.
- **Trains using SFTTrainer**, optimized for instruction-following models.
- **Pushes fine-tuned model to Hugging Face Hub**.
- **Performs inference on fine-tuned model**.

## 📂 Project Structure
```
📂 mistral-finetuning-samsum
│── train.py                 # Fine-tuning script
│── inference.py             # Running inference after fine-tuning
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── config/
│   ├── peft_config.json     # LoRA configuration
│   ├── training_args.json   # Training arguments
│── models/
│   ├── mistral-finetuned-samsum/  # Fine-tuned model (saved locally)
```

---
## 🛠 Installation
Ensure you have Python 3.8+ and install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

```
torch
transformers
accelerate
peft
bitsandbytes
trl
huggingface_hub
```

---
## 📊 Fine-Tuning Mistral-7B with LoRA
Run the following script to fine-tune the model:
```bash
python train.py
```

### **Fine-Tuning Script (`train.py`)**
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

# Load model with 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# Define LoRA configuration
peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="mistral-finetuned-samsum",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=1,
    max_steps=250,
    fp16=True,
    push_to_hub=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model, train_dataset=data, peft_config=peft_config,
    args=training_args, tokenizer=tokenizer
)

trainer.train()
trainer.push_to_hub()
```

---
## 🔍 Running Inference on the Fine-Tuned Model
After fine-tuning, you can generate summaries using:
```bash
python inference.py
```

### **Inference Script (`inference.py`)**
```python
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig, AutoTokenizer
import torch

# Load fine-tuned tokenizer
tokenizer = AutoTokenizer.from_pretrained("/content/mistral-finetuned-samsum")

# Load fine-tuned model with LoRA adapters
model = AutoPeftModelForCausalLM.from_pretrained("/content/mistral-finetuned-samsum")
model.eval()
model.to("cuda")

generation_config = GenerationConfig(
    max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9, do_sample=True
)

prompt = "Summarize the following conversation: Alice: Hey, how's your day? Bob: It's been great, I worked on a cool project."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, generation_config=generation_config)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---
## ☁️ Uploading to Hugging Face Hub
After training, push the model to Hugging Face Hub:
```python
from huggingface_hub import notebook_login
notebook_login()
trainer.push_to_hub()
```

Once uploaded, you can use the model anywhere:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "your-username/mistral-finetuned-samsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```

---
## 📌 Summary
✅ **Fine-tunes Mistral-7B using LoRA on Samsum dataset**.
✅ **Uses 4-bit quantization to fit on consumer GPUs**.
✅ **Efficient fine-tuning using `SFTTrainer`**.
✅ **Uploads model to Hugging Face Hub**.
✅ **Performs inference with the fine-tuned model**.

---
## 📝 Citation
If you use this fine-tuned model, please cite the original Mistral-7B model:
```
@misc{mistralai2023,
  title={Mistral 7B},
  author={Mistral AI},
  year={2023},
  url={https://mistral.ai/}
}
```

---
## 🎯 Next Steps
🔹 Deploy the model using **FastAPI** or **Gradio**.  
🔹 Train on custom datasets beyond Samsum.  
🔹 Experiment with **LoRA rank, dropout, and optimizer settings**.  

---

