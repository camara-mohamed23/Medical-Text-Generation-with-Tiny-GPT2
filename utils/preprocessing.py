# utils/preprocessing.py

import re

def clean_text(text):
    """
    Nettoyage de texte simple pour enlever caractères spéciaux et doublons
    """
    # Enlever les balises HTML s'il y en a
    text = re.sub(r'<.*?>', '', text)
    # Enlever les caractères spéciaux inutiles
    text = re.sub(r'[^a-zA-Z0-9À-ÿ\s,.?]', '', text)
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_file(input_file, output_file):
    """
    Lecture d'un fichier, nettoyage et sauvegarde
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = [clean_text(line) for line in lines if line.strip() != '']

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    print(f"Fichier nettoyé sauvegardé dans {output_file}")

if __name__ == "__main__":
    preprocess_file("data/raw_medical_text.txt", "data/medical_text.txt")
