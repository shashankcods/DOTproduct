import torch 
import torch.nn as nn 
import torch.nn.functional as F # for softmax() and argmax()
from torch.optim import AdamW 
from torch.utils.data import TensorDataset, DataLoader 

import lightning as L 

print("CUDA available: " + str(torch.cuda.is_available()))
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", str(device).upper())

with open("datasets/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

token_to_id = {ch:i for i,ch in enumerate(chars)}
id_to_token = {i:ch for i,ch in enumerate(chars)}

data = torch.tensor([token_to_id[c] for c in text], dtype=torch.long)
block_size = 128                                                           

batch_size = 32

def get_batch():
    
    ix = torch.randint(len(data) - block_size, (batch_size,)) # batch_size is the num of sequences being provided per get_batch
    
    x = torch.stack([data[i:i+block_size] for i in ix])     # creating a stack of "batch_size" sequences to send as X
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # shifting the above one pos ahead to send as Y
    
    return x, y

class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_len):
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

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads=8):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        B, T, C = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # split heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores
        sims = torch.matmul(q, k.transpose(-2, -1))
        sims = sims / (self.head_dim ** 0.5)

        if mask is not None:
            sims = sims.masked_fill(mask, -1e9)

        weights = F.softmax(sims, dim=-1)

        out = torch.matmul(weights, v)

        # combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.W_o(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(0.1)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask):

        norm_x = self.ln1(x)
        attn = self.attn(norm_x, mask=mask)
        x = x + self.dropout1(attn)

        norm_x = self.ln2(x)
        ff = self.ff(norm_x)
        x = x + self.dropout2(ff)

        return x

class DecoderOnlyTranformer(L.LightningModule):
    
    def __init__(self, num_tokens, d_model, max_len):
        super().__init__()

        L.seed_everything(seed = 42)

        self.we = nn.Embedding(num_embeddings = num_tokens, embedding_dim = d_model)
        self.pe = PositionEncoding(d_model = d_model, max_len = max_len)

        num_layers = 6

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads=8) for _ in range(num_layers)]
        )

        self.fc_layer = nn.Linear(in_features = d_model, out_features = num_tokens)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):

        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)
        
        T = token_ids.size(1)
        mask = torch.tril(torch.ones((T, T), device=self.device))
        mask = mask == 0
        mask = mask.unsqueeze(0).unsqueeze(0)

        x = position_encoded

        for block in self.blocks:
            x = block(x, mask)

        fc_layer_output = self.fc_layer(x)

        return fc_layer_output
    
def generate(model, prompt, max_new_tokens=200):

    model.eval()

    model_input = torch.tensor([token_to_id[c] for c in prompt]).unsqueeze(0).to(device)

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

    model = DecoderOnlyTranformer(num_tokens=vocab_size, d_model=256, max_len=block_size).to(device) # now runs on GPU

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01
    )

    max_steps = 30000

    for step in range(max_steps):

        x, y = get_batch()

        x = x.to(device) # training data also in GPU
        y = y.to(device)

        logits = model(x)

        loss = model.loss(
            logits.view(-1, logits.size(-1)),  # (32, 32, 65) becomes (1024, 65)
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