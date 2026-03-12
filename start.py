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
