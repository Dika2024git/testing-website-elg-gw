import os
import json
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TextDataset, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from .utils import clean_output

MODEL_NAME = "distilgpt2"
MODEL_PATH = "models/chatbot_model"
DATASET_JSON = "api/dataset.json"

def convert_dataset(json_file):
    txt_path = json_file.replace(".json", ".txt")
    if os.path.exists(txt_path):
        return txt_path
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in data:
            if "text" in item:
                f.write(item["text"].strip() + "\n")
    return txt_path

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    dataset_path = convert_dataset(DATASET_JSON)
    dataset = TextDataset(tokenizer=tokenizer, file_path=dataset_path, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_total_limit=1,
        logging_steps=10,
        logging_dir='./logs'
    )

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)
    trainer.train()
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    return tokenizer, model

def generate_smart_response(tokenizer, model, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.85,
        pad_token_id=tokenizer.eos_token_id
    )
    return clean_output(tokenizer.decode(output[0], skip_special_tokens=True))
