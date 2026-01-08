# scripts/train.py

import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers le fichier texte médical")
    parser.add_argument("--output_dir", type=str, default="./models/med-gpt", help="Dossier pour sauvegarder le modèle")
    parser.add_argument("--epochs", type=int, default=1, help="Nombre d'epochs")
    args = parser.parse_args()

    # Charger tokenizer et modèle GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Préparer le dataset
    train_dataset = load_dataset(args.data_path, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configurer l'entraînement
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    # Lancer l'entraînement
    trainer.train()

    # Sauvegarder modèle et tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Modèle entraîné et sauvegardé dans {args.output_dir}")

if __name__ == "__main__":
    main()
