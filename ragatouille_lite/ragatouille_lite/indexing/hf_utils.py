import os
import shutil
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HF_HOME
from huggingface_hub.utils import RepositoryNotFoundError


def get_repo_from_hub(repo_id: str):
    api = HfApi()
    local_path = os.path.join(HF_HOME, "ragatouille", repo_id)
    commit_hash_file = os.path.join(local_path, "commit_hash.txt")

    try:
        # Check if the repo exists on the hub
        repo_info = api.repo_info(repo_id)
    except RepositoryNotFoundError:
        print(f"Repository {repo_id} not found on the Hugging Face Hub.")
        return

    # Check if the repo exists locally
    if os.path.exists(local_path):
        if os.path.exists(commit_hash_file):
            with open(commit_hash_file, "r") as f:
                local_commit_hash = f.read().strip()

            # Get the latest commit hash from the hub
            latest_commit_hash = repo_info.sha

            if local_commit_hash != latest_commit_hash:
                print(f"Updating local repository for {repo_id}...")
                # Remove existing content
                for item in os.listdir(local_path):
                    item_path = os.path.join(local_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)

                # Download the latest content
                hf_hub_download(
                    repo_id=repo_id, local_dir=local_path, local_dir_use_symlinks=False
                )

                # Update the commit hash file
                with open(commit_hash_file, "w") as f:
                    f.write(latest_commit_hash)
            else:
                print(f"Local repository for {repo_id} is up to date.")
        else:
            print(
                f"commit_hash.txt not found. Updating local repository for {repo_id}..."
            )
            hf_hub_download(
                repo_id=repo_id, local_dir=local_path, local_dir_use_symlinks=False
            )
            with open(commit_hash_file, "w") as f:
                f.write(repo_info.sha)
    else:
        print(f"Downloading repository {repo_id}...")
        os.makedirs(local_path, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id, local_dir=local_path, local_dir_use_symlinks=False
        )
        with open(commit_hash_file, "w") as f:
            f.write(repo_info.sha)

    print(f"Repository {repo_id} is ready for use.")


def push_folder_to_hub(local_folder: str, repo_id: str, repo_type: str = "model"):
    """
    Push the entire content of a local folder to a Hugging Face repository.

    Args:
        local_folder (str): The local path to the folder to be pushed.
        repo_id (str): The Hugging Face repository ID (e.g., 'username/repo-name').
        repo_type (str): The type of repository. Defaults to "model".

    Raises:
        ValueError: If the folder doesn't exist or if there's an error during the upload.
    """
    from huggingface_hub import HfApi, Repository

    if not os.path.isdir(local_folder):
        raise ValueError(f"The specified folder '{local_folder}' does not exist.")

    try:
        # Initialize the Hugging Face API
        api = HfApi()

        # Create or clone the repository
        repo = Repository(
            local_dir=local_folder, clone_from=repo_id, repo_type=repo_type
        )

        # Add all files in the folder
        repo.git_add(auto_lfs_track=True)

        # Commit the changes
        repo.git_commit("Update repository content")

        # Push the changes to the hub
        repo.git_push()

        print(f"Successfully pushed the content of '{local_folder}' to '{repo_id}'.")
    except Exception as e:
        raise ValueError(f"Error pushing folder to hub: {str(e)}") from e
