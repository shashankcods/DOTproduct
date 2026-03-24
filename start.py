import torch 
import torch.nn as nn 
import torch.nn.functional as F # for softmax() and argmax()
from torch.optim import Adam 
from torch.utils.data import TensorDataset, DataLoader 

import lightning as L 

token_to_id = {
               'what' : 0,
               'is' : 1,
               'statquest' : 2,
               'awesome' : 3,
               '<EOS>' : 4,
               }

id_to_token = dict(map(reversed, token_to_id.items()))

inputs = torch.tensor([[token_to_id["what"], ## input #1: what is statquest <EOS> awesome
                        token_to_id["is"], 
                        token_to_id["statquest"], 
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]], 
                       
                       [token_to_id["statquest"], # input #2: statquest is what <EOS> awesome
                        token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"]]])

labels = torch.tensor([[token_to_id["is"], 
                        token_to_id["statquest"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"], 
                        token_to_id["<EOS>"]],  
                       
                       [token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["awesome"], 
                        token_to_id["<EOS>"]]])

dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)

class PositionEncoding(nn.module):

    def __init__(self, d_model = 2, max_len = 6):
        # d_model = dimension of transformer, number of embeddings per token
        # max_len = max length of phrases

        super().__init__()

        pe = torch.zeroes(max_len, d_model) # creates a matrix of zeroes
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
        return word_embeddings + self.pe[:word_embeddings.size(0), :]