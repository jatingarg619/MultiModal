import torch
from torchvision import datasets, transforms
from PIL import Image
import uuid
import json
import os
from tqdm import tqdm

# Import our SmolVLM2 setup
from demo import load_model

def process_local_image(image_path, question, model, processor):
    """Process a local image file"""
    image = Image.open(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def setup_directories():
    """Create necessary directories for dataset"""
    os.makedirs("cifar10_images", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

def get_questions():
    """Return list of questions to ask about each image"""
    return [
        "What is the main subject in this image?",
        "Can you describe the colors and patterns you see?",
        "What details can you notice in the background?",
        "What makes this image distinct from other similar images?",
        "Can you describe the positioning and arrangement of objects?",
        "What context or setting does this image suggest?",
        "Are there any interesting visual patterns or textures?",
        "How would you describe the lighting and contrast?",
        "What activities or actions are depicted, if any?",
        "How would you classify this image among standard image categories?"
    ]

def create_cifar10_dataset():
    """Create and save the dataset"""
    print("Setting up directories...")
    setup_directories()
    
    # First check if dataset file already exists
    output_path = "dataset/cifar10_test_conversations.json"
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        print("Do you want to create it again? This will overwrite the existing dataset.")
        response = input("Enter 'yes' to continue or any other key to exit: ")
        if response.lower() != 'yes':
            print("Exiting without creating new dataset.")
            return
    
    print("Loading CIFAR10 test dataset...")
    try:
        transform = transforms.ToTensor()
        cifar_dataset = datasets.CIFAR10(root='./data', train=False, 
                                        download=True, transform=transform)
        print(f"Successfully loaded CIFAR10 test dataset with {len(cifar_dataset)} images")
    except Exception as e:
        print(f"Error loading CIFAR10 dataset: {str(e)}")
        return
    
    print("Loading SmolVLM2 model...")
    try:
        model, processor = load_model()
        print("Successfully loaded SmolVLM2 model")
    except Exception as e:
        print(f"Error loading SmolVLM2 model: {str(e)}")
        return
    
    dataset = []
    questions = get_questions()
    
    # Create a subset of 100 images
    target_images = 100
    print(f"Processing {target_images} test images...")
    
    for idx, (image, label) in enumerate(tqdm(cifar_dataset, total=target_images)):
        if idx >= target_images:
            break
            
        # Convert to PIL Image
        pil_image = transforms.ToPILImage()(image)
        
        # Save image
        img_path = f"cifar10_images/test_image_{label}_{uuid.uuid4()}.jpg"
        pil_image.save(img_path)
        
        # Create conversation for each image
        conversations = []
        for question in questions:
            try:
                response = process_local_image(img_path, question, model, processor)
                conversations.extend([
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "assistant",
                        "value": response
                    }
                ])
            except Exception as e:
                print(f"Error processing question '{question}' for image {img_path}: {str(e)}")
                continue
        
        # Save to dataset
        entry = {
            "id": str(uuid.uuid4()),
            "image": img_path,
            "conversations": conversations
        }
        dataset.append(entry)
        
        # Save progress periodically
        if (idx + 1) % 10 == 0:
            save_dataset(dataset)
            print(f"Processed {idx + 1}/{target_images} images")
    
    # Final save
    save_dataset(dataset)
    print("Dataset creation completed!")

def save_dataset(dataset):
    """Save the dataset to a JSON file"""
    output_path = "dataset/cifar10_test_conversations.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    create_cifar10_dataset() 