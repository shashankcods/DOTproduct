from datasets import load_dataset
import json
import os

dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

os.makedirs("datasets/sharegpt", exist_ok=True)

out = open("datasets/sharegpt/sharegpt.jsonl", "w", encoding="utf-8")

for row in dataset["train_sft"]:

    conversation = row["messages"]

    for msg in conversation:

        role = msg["role"]
        text = msg["content"].strip()

        if role == "user":
            out.write(f"User: {text}\n")

        elif role == "assistant":
            out.write(f"Assistant: {text}\n")

    out.write("\n")

out.close()

print("ShareGPT dataset saved")