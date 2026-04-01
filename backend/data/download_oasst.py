from datasets import load_dataset
import json
import os

dataset = load_dataset("OpenAssistant/oasst1")

os.makedirs("datasets/oasst", exist_ok=True)

out = open("datasets/oasst/oasst.jsonl", "w", encoding="utf-8")

for row in dataset["train"]:

    role = row["role"]
    text = row["text"].strip()

    if role == "user":
        out.write(f"User: {text}\n")

    elif role == "assistant":
        out.write(f"Assistant: {text}\n")

    if role == "assistant":
        out.write("\n")

out.close()

print("OASST dataset saved")