from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

def upload_to_hub(
    model_path="vlm_model",
    repo_name="your-username/siglip-phi-mini-vlm",  # Change this to your username
    token=None  # You'll need to provide your HF token
):
    # Create the repo
    api = HfApi()
    
    try:
        create_repo(repo_name, private=False, token=token)
    except Exception as e:
        print(f"Repo might already exist: {e}")
    
    # Push the model to the hub
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Save model and tokenizer to hub
    model.push_to_hub(repo_name, token=token)
    tokenizer.push_to_hub(repo_name, token=token)
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import os
    # Get token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Please set the HF_TOKEN environment variable with your Hugging Face token")
        exit(1)
    
    upload_to_hub(token=hf_token) 