# Import required libraries for Hugging Face Hub interaction
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Define dataset repository details
# Replace <your-hf-username> with your Hugging Face username
repo_id = "panda1391/tourism-package-prediction"
repo_type = "dataset"

# Initialize API client using the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository already exists; create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created successfully.")

# Step 2: Upload the entire data folder to the Hugging Face dataset repository
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset uploaded to Hugging Face successfully.")
