from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

line_map = {}
with open("movie_lines.txt", "r", encoding="iso-8859-1") as f:
    for line in f:
        parts = line.strip().split(" +++$+++ ")
        if len(parts) == 5:
            line_map[parts[0]] = parts[4]

conversations = []
with open("movie_conversations.txt", "r", encoding="iso-8859-1") as f:
    for line in f:
        parts = line.strip().split(" +++$+++ ")
        if len(parts) == 4:
            line_ids = eval(parts[3])
            lines = [line_map.get(lid, "Unknown") for lid in line_ids]
            for i in range(len(lines) - 1):
                if "?" in lines[i]:
                    dialogue = f"Prompt: {lines[i]}\nResponse: {lines[i+1]}{tokenizer.eos_token}"
                    conversations.append(dialogue)

dataset = Dataset.from_dict({"text": conversations})

def preprocess_function(examples):
    encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings

tokenized_dataset = dataset.map(preprocess_function, batched=True)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./movie_dialogue_model",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=200,
    save_steps=1000,
    warmup_steps=1000,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    learning_rate=3e-5,
    weight_decay=0.1,
    gradient_accumulation_steps=8,
    fp16=True,
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./movie_dialogue_model_final")
tokenizer.save_pretrained("./movie_dialogue_model_final")
