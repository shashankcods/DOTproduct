import csv
import random
import re

train_path = "data/datasets/ECMB/train.csv"
val_path   = "data/datasets/ECMB/validation.csv"
test_path  = "data/datasets/ECMB/test.csv"

output_path = "data/datasets/ECMBconversations2.txt"


def clean_text(text):
    text = text.lower()

    text = text.replace("_comma_", ",")
    text = text.replace("_period_", ".")
    text = text.replace("_question_", "?")
    text = text.replace("_exclamation_", "!")

    text = re.sub(r"(\d+)%", r"\1 percent", text)
    
    text = re.sub(r"[^a-z0-9 .,!?'\n:<>_-]", "", text)

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def load_dataset(paths):
    conversations = {}

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                conv_id = row["conv_id"]
                utterance_idx = int(row["utterance_idx"])
                speaker_idx = row["speaker_idx"]
                utterance = clean_text(row["utterance"])

                if conv_id not in conversations:
                    conversations[conv_id] = []

                conversations[conv_id].append(
                    (utterance_idx, speaker_idx, utterance)
                )

    return conversations


def build_messages(rows):
    rows.sort(key=lambda x: x[0])

    speaker_order = []
    for _, speaker, _ in rows:
        if speaker not in speaker_order:
            speaker_order.append(speaker)

    if len(speaker_order) < 2:
        return None

    user_speaker = speaker_order[0]
    model_speaker = speaker_order[1]

    messages = []
    prev_role = None
    buffer = ""

    for _, speaker, text in rows:
        role = "user" if speaker == user_speaker else "model"

        if role == prev_role:
            buffer += " " + text
        else:
            if buffer:
                messages.append((prev_role, buffer))
            buffer = text
            prev_role = role

    if buffer:
        messages.append((prev_role, buffer))

    if len(messages) < 2:
        return None

    return messages


def generate_windows(messages):
    samples = []

    for i in range(2, len(messages) + 1, 2):
        sample = messages[:i]
        samples.append(sample)

    return samples


def main():
    paths = [train_path, val_path, test_path]

    conversations = load_dataset(paths)

    training_samples = []

    for conv_id in conversations:
        rows = conversations[conv_id]

        messages = build_messages(rows)
        if messages is None:
            continue

        samples = generate_windows(messages)

        training_samples.extend(samples)

    random.shuffle(training_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in training_samples:
            f.write("<start_convo>\n")

            for role, text in sample:
                f.write(f"{role}: {text}\n")

            f.write("<end_convo>\n\n")


if __name__ == "__main__":
    main()