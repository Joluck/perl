import torch
import os
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(base_model_path: str, checkpoint_path: str, output_path: str = None, dtype: str = "bfloat16"):
    if output_path is None:
        output_path = os.path.join(checkpoint_path, "merged")

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"Loading adapter from {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    print("Merging adapter into base model")
    merged_model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PEFT adapter checkpoint into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the adapter checkpoint (e.g. output_dir/checkpoint-64)")
    parser.add_argument("--output", type=str, default=None, help="Output path for merged model (default: {checkpoint}/merged)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    merge(args.base_model, args.checkpoint, args.output, args.dtype)
