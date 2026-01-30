# train_lora_ranker.py
import argparse, json, torch, os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

# ----------------------- Args -----------------------
def get_args():
    ap = argparse.ArgumentParser(description="LoRA training for Danish address normalization (rank + format)")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--val_file",   default="data/val.jsonl")
    ap.add_argument("--out_dir",    default="lora-addr-normalizer")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--eval_steps", type=int, default=500)  # Kept for compatibility, but not used
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 compute")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--resume_from", default=None)
    return ap.parse_args()

# ----------------------- Data -----------------------
def load_jsonl_splits(train_file, val_file):
    data_files = {"train": train_file, "validation": val_file}
    ds = load_dataset("json", data_files=data_files)
    return ds["train"], ds["validation"]

def build_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=os.environ.get('HF_HOME'))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def make_formatter(tok, max_len: int):
    eos = tok.eos_token

    def _tokenize(example):
        # Expect keys: prompt, response
        prompt = example["prompt"].rstrip() + "\n"
        response = example["response"].rstrip() + eos

        # Tokenize separately to mask prompt
        tok_prompt = tok(prompt, add_special_tokens=False)
        tok_resp   = tok(response, add_special_tokens=False)

        input_ids = tok_prompt["input_ids"] + tok_resp["input_ids"]
        attention = [1] * len(input_ids)

        # Labels: mask prompt tokens with -100, learn on response only
        labels = [-100] * len(tok_prompt["input_ids"]) + tok_resp["input_ids"]

        # Truncate (keep the tail so we always keep response)
        if len(input_ids) > max_len:
            # We must keep entire response. Compute how many prompt tokens we can keep.
            resp_len = len(tok_resp["input_ids"])
            keep_prompt = max(0, max_len - resp_len)
            input_ids = tok_prompt["input_ids"][:keep_prompt] + tok_resp["input_ids"]
            attention = [1] * len(input_ids)
            labels = [-100] * keep_prompt + tok_resp["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
        }
    return _tokenize

# ----------------------- Data collator -----------------------
@dataclass
class DataCollatorForCausal:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # pad to max len in batch
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            if max_len % multiple != 0:
                max_len = ((max_len // multiple) + 1) * multiple

        input_ids, attn, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id
        for f in features:
            L = len(f["input_ids"])
            pad = max_len - L
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ----------------------- Model -----------------------
def load_lora_model(model_name, args, hf_home):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=bnb_config, cache_dir=hf_home
    )
    base = prepare_model_for_kbit_training(base)

    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    return model

# ----------------------- Main -----------------------
def main():
    args = get_args()
    torch.manual_seed(args.seed)

    # Set Hugging Face cache to scratch directory if available
    scratch_dir = os.environ.get('SCRATCH') or os.path.join('/', 'work3', 's232692')  # Use your scratch path
    hf_home = os.path.join(scratch_dir, 'hf_cache')
    os.makedirs(hf_home, exist_ok=True)
    os.environ['HF_HOME'] = hf_home
    print(f"Using Hugging Face cache at: {hf_home}")

    # Data
    train_ds, val_ds = load_jsonl_splits(args.train_file, args.val_file)
    tok = build_tokenizer(args.base_model)
    formatter = make_formatter(tok, args.max_len)

    train_tok = train_ds.map(formatter, remove_columns=train_ds.column_names, desc="Tokenizing train", num_proc=0, cache_file_name=os.path.join(hf_home, 'train_cache.arrow'))
    # val_tok not needed without evaluation
    # val_tok   = val_ds.map(formatter,   remove_columns=val_ds.column_names,   desc="Tokenizing val",   num_proc=1, cache_file_name=os.path.join(hf_home, 'val_cache.arrow'))

    collator = DataCollatorForCausal(tok)

    # Model
    model = load_lora_model(args.base_model, args, hf_home)

    # Train args
    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        eval_strategy="no",  # Disable evaluation to save GPU memory
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        # eval_dataset=val_tok,  # Removed to save GPU memory
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved LoRA adapter and tokenizer to", args.out_dir)

if __name__ == "__main__":
    main()
