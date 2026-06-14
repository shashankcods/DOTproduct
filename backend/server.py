from fastapi import FastAPI
from pydantic import BaseModel
import torch

from start import DecoderOnlyTranformer, generate, tokenizer, vocab_size, block_size, device

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str


model = DecoderOnlyTranformer(
    num_tokens=vocab_size,
    d_model=384,
    max_len=block_size
).to(device)

model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

print("Model loaded and ready!")

@app.post("/generate")
def generate_text(request: PromptRequest):
    output = generate(model, request.prompt)
    return {"response": output}