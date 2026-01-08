# fine_tune.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")

# Dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/medical_text_clean.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./models/tiny_gpt2_medical",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("./models/tiny_gpt2_medical")
print("✅ Fine-tuning terminé")
