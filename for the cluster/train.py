import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- CONFIGURATION ---
# Paths are relative to the project root
DATA_PATH = "data/train_challenger.jsonl"
OUTPUT_DIR = "lora_challenger_model"
MAX_SEQ_LENGTH = 2048

# 1. LOAD BASE MODEL
# We use the 4-bit quantized version for speed and memory efficiency
print(f"Loading Llama-3-8B model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-instruct-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto-detect
    load_in_4bit = True,
)

# 2. ADD LoRA ADAPTERS
# This setup targets all linear layers for maximum reasoning capability
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,             # Rank 32 is strong for reasoning tasks
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,   # Set to 0 for optimized training
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 3. LOAD & FORMAT DATA
print(f"Loading data from {DATA_PATH}...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. TRAINING
print("Starting training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 4, # Increase to 8 if A100 has 80GB VRAM
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 400,                 # ~1 epoch for 15k rows (adjust to 800 for 2 epochs)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "checkpoints",      # Temporary checkpoints
    ),
)

trainer.train()

# 5. SAVE MODEL
print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training Complete!")