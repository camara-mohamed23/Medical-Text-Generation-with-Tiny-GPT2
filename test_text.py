with open("data/medical_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("✅ Nombre de caractères :", len(text))
print("✅ Aperçu du texte :")
print(text[:300])
