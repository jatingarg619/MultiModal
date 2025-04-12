import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class CIFARDescriptionDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Get description (concatenate all assistant responses)
        descriptions = []
        for conv in item['conversations']:
            if conv['from'] == 'assistant':
                descriptions.append(conv['value'])
        description = " ".join(descriptions)
        
        return image, description 