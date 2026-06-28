# ============================================================
# Hosting Script — Push Deployment Files to Hugging Face Space
# ============================================================
# This script uploads the deployment folder (Dockerfile, app.py,
# requirements.txt) to the Hugging Face Space for the Streamlit app.

from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Replace <your-hf-username> with your Hugging Face username
# Space name must use hyphens (not underscores) to avoid API URL issues
api.upload_folder(
    folder_path="tourism_project/deployment",   # local folder with app files
    repo_id="panda1391/tourism-package-prediction",  # HF Space
    repo_type="space",
    path_in_repo="",  # upload to root of the space
)
print("Deployment files uploaded to Hugging Face Space successfully.")
