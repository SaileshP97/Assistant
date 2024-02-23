import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tokenizer = GPT2Tokenizer.from_pretrained("Tokenizer/")


def infer(inp, model):
    inp = "You " + inp + " Assistant"
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"]
    a = inp["attention_mask"]
    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])
    return output


model = GPT2LMHeadModel.from_pretrained("trained_model/")

print("Assistant: Ask anything...")
while True:
    text = str(input())
    if text == "exit":
        break
    output = infer(text, model)
    print(output)
