from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import Dataset
from tqdm import tqdm

def prepare_qlora_training(siglip_data_dir, output_dir="vlm_model"):
    # Load the same Phi-3 configuration as used in SigLIP's text encoder
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",  # Same model as SigLIP
        trust_remote_code=True,
        torch_dtype=torch.float32,
        load_in_4bit=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",  # Match the model
        trust_remote_code=True
    )
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA with target modules matching Phi-3's architecture
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Updated for Phi-3
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_vlm(siglip_data_dir, output_dir="vlm_model", batch_size=4, num_epochs=3):
    # Initialize dataset
    dataset = VLMDataset(siglip_data_dir)
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_qlora_training(siglip_data_dir, output_dir)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        warmup_ratio=0.1,
    )
    
    # Custom training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(range(len(dataset) // batch_size))
        
        for i in range(0, len(dataset), batch_size):
            batch_samples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            
            # Prepare inputs
            input_texts = [sample['input_text'] for sample in batch_samples]
            target_texts = [sample['text'] for sample in batch_samples]
            image_embeddings = torch.stack([sample['image_embeddings'] for sample in batch_samples])
            
            # Tokenize
            inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
            targets = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            targets = targets.input_ids.to(model.device)
            image_embeddings = image_embeddings.to(model.device)
            
            # Forward pass
            outputs = model(
                **inputs,
                labels=targets,
                image_embeddings=image_embeddings  # Custom forward pass handling in model
            )
            
            loss = outputs.loss
            loss.backward()
            
            if (i // batch_size + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.update(1)
            
        # Save checkpoint
        model.save_pretrained(f"{output_dir}/checkpoint-epoch-{epoch}")
        
        avg_loss = total_loss / (len(dataset) // batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model 