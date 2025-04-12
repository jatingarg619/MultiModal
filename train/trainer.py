import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def train_siglip(model, train_dataset, val_dataset=None, 
                 batch_size=8,  # Reduced batch size
                 num_epochs=10, 
                 learning_rate=1e-4,
                 gradient_accumulation_steps=4):  # Add gradient accumulation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create data loader with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)  # Reduced num_workers
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (images, texts) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits = model(images, texts)
                labels = torch.arange(len(images)).to(device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}") 