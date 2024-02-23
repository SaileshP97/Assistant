from transformers import GPT2LMHeadModel, GPT2Tokenizer

checkpoint = "gpt2"

model = GPT2LMHeadModel.from_pretrained(checkpoint)
model.save_pretrained("Model")

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
tokenizer.add_special_tokens(
    {"pad_token": "[PAD]", "bos_token": "\nYou", "eos_token": "[EOS]"}
)
tokenizer.add_tokens(["\nAssistant"])
tokenizer.save_pretrained("Tokenizer/")
