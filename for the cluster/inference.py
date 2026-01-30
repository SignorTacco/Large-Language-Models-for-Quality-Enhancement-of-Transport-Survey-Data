import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os

# --- CONFIGURATION ---
# OLD:
# INPUT_CSV = "data/full_200k_dataset_with_prompts.csv"

# NEW:
INPUT_CSV = "data/full_200k_dataset_RAG_READY.csv"

OUTPUT_CSV = "final_thesis_results.csv"
LORA_PATH = "lora_challenger_model" 

def main():
    # 1. LOAD DATA
    print(f"Loading data from {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
        
    df = pd.read_csv(INPUT_CSV)

    if 'prompt_text' not in df.columns:
        raise ValueError("The input CSV is missing the 'prompt_text' column!")

    prompts = df['prompt_text'].tolist()
    print(f"Loaded {len(prompts)} prompts.")

    # 2. INITIALIZE vLLM ENGINE
    print("Initializing vLLM engine...")
    # NOTE: tensor_parallel_size=1 is standard for single GPU.
    llm = LLM(
        model="unsloth/llama-3-8b-instruct-bnb-4bit", 
        enable_lora=True,
        tensor_parallel_size=1, 
        max_lora_rank=32
    )

    # 3. DEFINE SAMPLING PARAMS
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=128,
        stop=["<|end_of_text|>"]
    )

    # 4. RUN INFERENCE
    print("Running batch inference...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("thesis_adapter", 1, LORA_PATH)
    )

    # 5. PARSE & SAVE RESULTS
    print("Parsing results...")
    generated_texts = [output.outputs[0].text for output in outputs]

    df['llm_full_response'] = generated_texts

    # Extract clean VALID/INVALID verdict
    df['llm_verdict'] = df['llm_full_response'].apply(
        lambda x: "INVALID" if "INVALID" in x.upper() else "VALID"
    )

    print(f"Saving results to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Inference Complete!")

# --- THIS IS THE CRITICAL FIX ---
if __name__ == "__main__":
    main()