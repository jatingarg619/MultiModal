import torch
import json
import numpy as np
from torch.utils.data import Dataset

class VLMDataset(Dataset):
    def __init__(self, siglip_data_dir):
        # Load metadata and processed dataset
        with open(f"{siglip_data_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        with open(f"{siglip_data_dir}/processed_dataset.json", 'r') as f:
            self.data = json.load(f)
            
        # Template for instruction format
        self.instruction_template = """Below is an image. Please describe it in detail.

Image: <image>
Description:"""

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load embeddings
        embeddings = np.load(f"siglip_processed_data/sample_{idx}_embeddings.npz")
        
        # Format input with instruction
        formatted_input = self.instruction_template.replace("<image>", "[IMAGE]")
        
        return {
            'image_embeddings': torch.tensor(embeddings['image_embedding']),
            'text': sample['text'],
            'input_text': formatted_input,
            'label': sample['label']
        }

    def __len__(self):
        return len(self.data) 