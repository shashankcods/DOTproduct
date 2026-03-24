import pandas as pd
import re
import os

RAW_PATH = "backend/datasets/raw/"
PROCESSED_PATH = "backend/datasets/processed/"

os.makedirs(PROCESSED_PATH, exist_ok=True)


import re

def clean_text(text):
    import re

import re

def clean_text(text):
    # Remove speaker names
    text = re.sub(r"[A-Za-z ]+:", "", text)

    # Lowercase
    text = text.lower()

    # Keep punctuations
    text = re.sub(r"[^a-z0-9\s\.\,\!\?']", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    # Extract sentences with punctuation
    sentences = re.findall(r"[^.!?]+[.!?]", text)

    # Clean sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    return "\n".join(sentences)


def process_file(filename):
    print(f"Processing {filename}...")

    df = pd.read_csv(RAW_PATH + filename)

    if "text" in df.columns:
        text_data = " ".join(df["text"].astype(str))
    else:
        # fallback
        text_data = " ".join(df.iloc[:, 0].astype(str))

    cleaned = clean_text(text_data)

    output_file = filename.replace(".csv", ".txt")
    with open(PROCESSED_PATH + output_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Saved → {output_file}\n")


if __name__ == "__main__":
    process_file("train.csv")
    process_file("validation.csv")
    process_file("test.csv")

    print("All files processed")