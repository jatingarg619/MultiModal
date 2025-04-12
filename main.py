from model.siglip import SigLIPModel
from train.dataset import CIFARDescriptionDataset
from train.trainer import train_siglip
from torchvision import transforms

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = CIFARDescriptionDataset(
        json_path='dataset/cifar10_test_conversations.json',
        transform=transform
    )
    
    # Create model
    model = SigLIPModel()
    
    # Train model
    train_siglip(
        model=model,
        train_dataset=dataset,
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4
    )

if __name__ == "__main__":
    main() 