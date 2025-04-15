import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def fine_tune_model(dataset_path, output_dir, base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Fine-tune TinyLlama with QLoRA"""
    print(f"📂 Dataset path: {dataset_path}")
    print(f"📦 Loading base model: {base_model}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Trainable parameters: {trainable:,}")

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    print(f"📊 Loaded {len(dataset)} training samples")

    # Remove bad samples (missing fields)
    dataset = dataset.filter(lambda x: x.get("instruction") and x.get("output"))

    def format_prompt(example):
        prompt = f"<|im_start|>user\n{example['instruction']}"
        if example.get("input"):
            prompt += f"\n{example['input']}"
        prompt += f"<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": prompt}

    dataset = dataset.map(format_prompt)

    def tokenize(example):
        return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("🧠 Starting training...")
    trainer.train()

    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✅ Training complete! Model saved to {final_model_dir}")

    return final_model_dir


if __name__ == "__main__":
    print("🔍 Checking CUDA availability...")
    if torch.cuda.is_available():
        print("🚀 GPU:", torch.cuda.get_device_name(0))
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "data", "processed_financial_data.json")
    output_dir = os.path.join(current_dir, "models", "fine_tuned")
    os.makedirs(output_dir, exist_ok=True)

    try:
        fine_tuned_model = fine_tune_model(dataset_path, output_dir)
    except Exception as e:
        print(f"❌ Fine-tuning failed: {str(e)}")
        print("💡 Suggestions:")
        print("- Ensure the dataset exists and follows the correct structure.")
        print("- Try reducing batch size or model size if facing OOM errors.")
        print("- Check for compatibility between Transformers and PEFT versions.")
