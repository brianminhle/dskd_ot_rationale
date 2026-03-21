from huggingface_hub import snapshot_download

# Repository ID and local directory
repo_id = "openai-community/gpt2-xl"
local_dir = "/mnt/bn/magellan-product-llm-data/tu.vu/matrix_one/dskd_ot_rationale/model_hub/gpt2"  # Specify your desired local directory

# Download the model repository
print(f"Downloading model from {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="model",  # Specify repo type as 'model'
    local_dir=local_dir,  # Specify the local directory
    local_dir_use_symlinks=False,  # Avoid symlinks for complete local copy
)

print(f"Model successfully downloaded to {local_dir}!")