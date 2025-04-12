import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def train_siglip(model, train_dataset, val_dataset=None, 
                 batch_size=32, num_epochs=10, learning_rate=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, texts) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            
            # Forward pass
            logits = model(images, texts)
            
            # Compute sigmoid loss
            labels = torch.arange(len(images)).to(device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}") 