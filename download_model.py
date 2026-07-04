from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def download_t5_base(save_directory=None):
    model_name = "t5-base"
    
    print(f"Downloading {model_name}...")
    
    # Download Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if save_directory:
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        print(f"Saving to {save_directory}...")
        tokenizer.save_pretrained(save_directory)
        model.save_pretrained(save_directory)
        print("Download and save complete.")
    else:
        print("Download complete (saved to default Hugging Face cache).")

if __name__ == "__main__":
    # Option 1: Save to default cache
    # download_t5_base()
    
    # Option 2: Save to a specific folder (Useful for clusters like Snellius)
    # This matches the structure you used in your bash script ($ENV_PATH/local_models/...)
    target_path = "local_models/google/t5-base" 
    download_t5_base(target_path)
