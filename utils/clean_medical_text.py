# utils/clean_medical_text.py
with open("data/medical_text.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Supprimer doublons et lignes trop courtes
lines = list({line.strip() for line in lines if len(line.strip()) > 20})

with open("data/medical_text_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ Texte nettoyé enregistré dans data/medical_text_clean.txt")
