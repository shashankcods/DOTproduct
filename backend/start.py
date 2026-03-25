import torch 
import torch.nn as nn 
import torch.nn.functional as F # for softmax() and argmax()
from torch.optim import Adam 
from torch.utils.data import TensorDataset, DataLoader 

import lightning as L 

with open("datasets/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

token_to_id = {ch:i for i,ch in enumerate(chars)}
id_to_token = {i:ch for i,ch in enumerate(chars)}

data = torch.tensor([token_to_id[c] for c in text], dtype=torch.long)
block_size = 32                                                              # can be increased later

batch_size = 32

def get_batch():
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

class PositionEncoding(nn.Module):

    def __init__(self, d_model = 2, max_len = 6):
        # d_model = dimension of transformer, number of embeddings per token
        # max_len = max length of phrases

        super().__init__()

        pe = torch.zeros(max_len, d_model) # creates a matrix of zeroes
        position = torch.arange(start = 0, end = max_len, step = 1).float().unsqueeze(1) # unsqueeze list into matrix

        embedding_index = torch.arange(start = 0, end = d_model, step = 2).float() # embedding index used is 2i anyway
        div_term = 1/torch.tensor(100000.0)**(embedding_index/d_model)

        # position encoding equations (from "Attention is all you need")
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # every alt column has sin values, starting with the 1st
        pe[:, 1::2] = torch.cos(position * div_term) # the same but w cos and from 2nd

        self.register_buffer('pe', pe) # so that not treated as a parameter and moves with the model
    
    def forward(self, word_embeddings):
        seq_len = word_embeddings.size(1)
        return word_embeddings + self.pe[:seq_len, :].unsqueeze(0)

class Attention(nn.Module):

    def __init__(self, d_model = 2):
        super().__init__()

        self.d_model = d_model

        self.W_q = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask = None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(-2, -1))

        scaled_sims = sims / (k.size(-1) ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask, -1e9)

        attention_percents = F.softmax(scaled_sims, dim=-1)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

class DecoderOnlyTranformer(L.LightningModule):
    
    def __init__(self, num_tokens = 4, d_model = 2, max_len = 6):
        super().__init__()

        L.seed_everything(seed = 42)

        self.we = nn.Embedding(num_embeddings = num_tokens, embedding_dim = d_model)
        self.pe = PositionEncoding(d_model = d_model, max_len = max_len)

        self.self_attention = Attention(d_model = d_model)

        self.fc_layer = nn.Linear(in_features = d_model, out_features = num_tokens)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):

        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)
        
        T = token_ids.size(1)
        mask = torch.tril(torch.ones((T, T), device=self.device))
        mask = mask == 0 # converting mask into bool values from 0/1
        mask = mask.unsqueeze(0) # to add an extra dimension for batching

        self_attention_values = self.self_attention(position_encoded, position_encoded, position_encoded, mask = mask)

        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output
    
def generate(model, prompt, max_new_tokens=200):

    model.eval()

    model_input = torch.tensor([token_to_id[c] for c in prompt]).unsqueeze(0)

    generated = []

    for _ in range(max_new_tokens):

        model_input = model_input[:, -block_size:]

        predictions = model(model_input)

        probs = F.softmax(predictions[0, -1], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        model_input = torch.cat(
            (model_input, next_id.unsqueeze(0)),
            dim=1
        )

        generated.append(id_to_token[next_id.item()])

    return prompt + "".join(generated)

if __name__ == "__main__":

    model = DecoderOnlyTranformer(num_tokens=vocab_size, d_model=32, max_len=block_size)

    optimizer = Adam(model.parameters(), lr=3e-4)

    max_steps = 10000

    for step in range(max_steps):

        x, y = get_batch()

        logits = model(x)

        loss = model.loss(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print("step:", step, "loss:", loss.item())

    prompt = "ROMEO:"
    output = generate(model, prompt)

    print(output)