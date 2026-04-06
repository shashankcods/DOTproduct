from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["data/datasets/ECMBconversations2.txt"],
    vocab_size=16000,
    min_frequency=2
)

tokenizer.save_model("tokenizer")