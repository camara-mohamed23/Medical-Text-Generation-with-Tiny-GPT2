from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "sshleifer/tiny-gpt2"
DATA_PATH = "data/medical_text_clean.txt"
OUTPUT_DIR = "models/tiny_gpt2_medical"
EPOCHS = 3
LR = 5e-5
MAX_LENGTH = 512

# ==============================
# DOSSIER DE SORTIE
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# CHARGEMENT MODELE & TOKENIZER
# ==============================
print("🔹 Chargement du modèle Tiny-GPT2...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME,
    use_safetensors=True   # ✅ OBLIGATOIRE (macOS / torch < 2.6)
)

# GPT2 n'a pas de pad_token par défaut
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# ==============================
# LECTURE DU TEXTE MEDICAL
# ==============================
print("🔹 Lecture du texte médical...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenisation
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH
)

# ==============================
# OPTIMISEUR
# ==============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ==============================
# ENTRAINEMENT
# ==============================
print("🚑 Entraînement médical en cours (CPU)...")
model.train()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"]
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"✅ Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item():.4f}")

# ==============================
# SAUVEGARDE
# ==============================
print("💾 Sauvegarde du modèle médical...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Fine-tuning terminé avec succès !")
