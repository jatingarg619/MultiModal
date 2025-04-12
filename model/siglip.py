import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SigLIPModel(nn.Module):
    def __init__(self, phi_model_name="microsoft/Phi-3-mini-4k-instruct", image_size=32):
        super().__init__()
        
        # Image encoder (for CIFAR10 images: 32x32x3)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512)
        )
        
        # Text encoder (Phi-3)
        self.text_tokenizer = AutoTokenizer.from_pretrained(phi_model_name)
        self.text_encoder = AutoModel.from_pretrained(phi_model_name)
        
        # Project text embeddings to same dimension as image embeddings
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, 512)
        
        # Temperature parameter for loss scaling
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def encode_image(self, image):
        return self.image_encoder(image)
    
    def encode_text(self, text, device):
        tokenized = self.text_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Move tokenized inputs to the correct device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        text_features = self.text_encoder(**tokenized).last_hidden_state[:, 0, :]  # Take [CLS] token
        return self.text_projector(text_features)
    
    def forward(self, image, text):
        # Get embeddings
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, image.device)
        
        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        
        # Compute similarity
        logits = torch.matmul(image_features, text_features.T) * torch.exp(self.temperature)
        
        return logits 