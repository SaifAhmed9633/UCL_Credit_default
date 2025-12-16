from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)

ds = load_dataset("json", data_files="train.jsonl")["train"]

def fmt(x): return x["prompt"] + x["completion"]

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    formatting_func=fmt,
    args=TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=50,
        fp16=True,
    ),
)

trainer.train()
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
print("Saved: lora_adapter/")
