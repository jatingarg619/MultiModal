class VLMDataset(torch.utils.data.Dataset):
    def __init__(self, processed_data_dir):
        # Load metadata
        with open(f"{processed_data_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
            
        # Load processed dataset
        with open(f"{processed_data_dir}/processed_dataset.json", 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load embeddings
        embeddings = np.load(f"{self.metadata['processed_data_dir']}/sample_{idx}_embeddings.npz")
        
        return {
            'image_embedding': torch.tensor(embeddings['image_embedding']),
            'text_embedding': torch.tensor(embeddings['text_embedding']),
            'text': sample['text'],
            'label': sample['label']
        } 