# scripts/generate.py

import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_text(prompt, model_dir, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models/med-gpt", help="Dossier du modèle entraîné")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour générer du texte")
    args = parser.parse_args()

    output_text = generate_text(args.prompt, args.model_dir)
    print("\n--- Texte généré ---")
    print(output_text)

if __name__ == "__main__":
    main()
