from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("🔹 Chargement du tokenizer et du modèle...")

tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")

# ⚡ Utiliser safetensors pour éviter l'erreur torch < 2.6
model = GPT2LMHeadModel.from_pretrained(
    "sshleifer/tiny-gpt2",
    use_safetensors=True  # <- clé ici
)

print("✅ Modèle et tokenizer chargés")

prompt = "Le diabète est une maladie"
inputs = tokenizer(prompt, return_tensors="pt")

print("🔹 Lancement de la génération (CPU rapide)...")

outputs = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50
)

print("🩺 Réponse du modèle :")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
