# med_gpt.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def train_med_gpt(dataset_path, output_dir="./med-gpt", epochs=1):
    # Charger le tokenizer et le modèle GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Préparer le dataset
    train_dataset = load_dataset(dataset_path, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50
    )

    # Entraînement
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Modèle entraîné et sauvegardé !")

def generate_text(prompt, model_dir="./med-gpt", max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_k=50)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    # Chemin vers ton fichier texte médical
    dataset_path = "medical_text.txt"  # chaque phrase médicale sur une ligne
    train_med_gpt(dataset_path, epochs=1)

    # Tester le modèle
    prompt = "Qu'est-ce que le diabète ?"
    print(generate_text(prompt))
