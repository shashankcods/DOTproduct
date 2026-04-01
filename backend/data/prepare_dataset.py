import os

output_path = "datasets/conversations.txt"

out = open(output_path, "w", encoding="utf-8")

# ShareGPT
with open("datasets/sharegpt/sharegpt.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        out.write(line)

# OASST
with open("datasets/oasst/oasst.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        out.write(line)

out.close()

print("Combined dataset saved to conversations.txt")