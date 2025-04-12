import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import json
import os

class CIFARDescriptionDataset(Dataset):
    def __init__(self, json_path, transform=None):
        # Load our JSON descriptions
        with open(json_path, 'r') as f:
            self.descriptions_data = json.load(f)
            
        # Download CIFAR10 test dataset
        self.cifar_dataset = datasets.CIFAR10(
            root='./data', 
            train=False,  # Using test set as per our previous setup
            download=True
        )
        
        self.transform = transform
        
    def __len__(self):
        return len(self.descriptions_data)
    
    def __getitem__(self, idx):
        item = self.descriptions_data[idx]
        
        # Extract image index and label from our saved image path
        # Assuming image path format: "cifar10_images/image_{label}_{uuid}.jpg"
        image_path = item['image']
        label = int(image_path.split('_')[2])  # Get the label from filename
        
        # Get corresponding CIFAR10 image
        cifar_images_with_label = [(i, img) for i, (img, l) in enumerate(self.cifar_dataset) if l == label]
        if cifar_images_with_label:
            _, image = cifar_images_with_label[0]  # Take the first image with matching label
        else:
            raise ValueError(f"No CIFAR10 image found with label {label}")
        
        # Convert to PIL Image and apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get description (concatenate all assistant responses)
        descriptions = []
        for conv in item['conversations']:
            if conv['from'] == 'assistant':
                descriptions.append(conv['value'])
        description = " ".join(descriptions)
        
        return image, description 