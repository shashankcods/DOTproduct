import pandas as pd
import os

train_path = "data/datasets/ECMB/train.csv"
val_path   = "data/datasets/ECMB/validation.csv"
test_path  = "data/datasets/ECMB/test.csv"

output_path = "data/datasets/ECMBconversations.txt"

print("Loading CSV files...")

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

df = pd.concat([df_train, df_val, df_test], ignore_index=True)

print("Total rows:", len(df))

df["utterance"] = df["utterance"].astype(str)
df["utterance"] = df["utterance"].str.replace("_comma_", ",", regex=False)

# sort conversation order
df = df.sort_values(["conv_id", "utterance_idx"])

print("Reconstructing conversations...")

conversations = []

for conv_id, group in df.groupby("conv_id"):

    group = group.sort_values("utterance_idx")

    convo_lines = []

    for _, row in group.iterrows():

        speaker_idx = int(row["speaker_idx"])

        # simple role mapping
        speaker = "User" if speaker_idx % 2 == 1 else "Assistant"

        text = row["utterance"].strip()

        if len(text) == 0:
            continue

        convo_lines.append(f"{speaker}: {text}")

    if len(convo_lines) > 1:
        conversation = "\n".join(convo_lines)
        conversations.append(conversation)

print("Total conversations:", len(conversations))

print("Writing dataset...")

with open(output_path, "w", encoding="utf-8") as f:

    for convo in conversations:
        f.write("<CONVO_START>\n")
        f.write(convo)
        f.write("\n<CONVO_END>\n\n")

print("Done.")
print("Saved to:", output_path)