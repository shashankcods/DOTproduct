from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

vocab = tokenizer.get_vocab()

# reverse mapping
id_to_token = {v:k for k,v in vocab.items()}

print("Vocab size:", len(vocab))
print()

print("Checking suspicious tokens...\n")

for token in id_to_token.values():
    if any(x in token for x in ["john", "ell", "ounce", "tok", "!!!"]):
        print(token)