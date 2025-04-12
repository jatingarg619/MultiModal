import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def train_siglip(model, train_dataset, val_dataset=None, 
                 batch_size=4, num_epochs=10, learning_rate=1e-4,
                 gradient_accumulation_steps=2):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')  # Updated syntax
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (images, texts) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):  # Updated syntax
                logits = model(images, texts)
                labels = torch.arange(len(images)).to(device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}") 