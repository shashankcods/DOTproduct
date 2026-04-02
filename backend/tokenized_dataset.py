from tokenizers import ByteLevelBPETokenizer
import numpy as np

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

input_file = "data/datasets/conversations.txt"
output_file = "data/tokens.bin"

chunk_size = 1_000_000  # characters per chunk

with open(input_file, "r", encoding="utf-8") as f, open(output_file, "wb") as out:

    while True:

        text = f.read(chunk_size)

        if not text:
            break

        tokens = tokenizer.encode(text).ids
        arr = np.array(tokens, dtype=np.uint16)

        arr.tofile(out)

print("Tokenization complete.")