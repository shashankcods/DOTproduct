import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)

from start import DecoderOnlyTranformer, generate, tokenizer, vocab_size, block_size, device

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
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

model.load_state_dict(
    torch.load(BACKEND_DIR / "model_weights.pth", map_location=device)
)
model.eval()

print("Model loaded and ready!")


@app.post("/generate")
def generate_text(request: PromptRequest):
    output = generate(model, request.prompt)
    return {"response": output}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)