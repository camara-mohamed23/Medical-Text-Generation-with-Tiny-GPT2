# test_model.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Charger tokenizer et modèle GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("✅ Modèle chargé")
print("Nombre de paramètres :", model.num_parameters())
