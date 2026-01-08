from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

print("🔹 Chargement du tokenizer Tiny-GPT2...")
tokenizer = GPT2Tokenizer.from_pretrained("models/tiny_gpt2_medical")

print("🔹 Chargement du modèle médical fine-tuné...")
model = GPT2LMHeadModel.from_pretrained("models/tiny_gpt2_medical")

model.eval()

# Texte de test
prompt = "Le diabète est une maladie"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🩺 Réponse du modèle :\n")
print(result)
