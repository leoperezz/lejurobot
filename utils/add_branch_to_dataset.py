#!/usr/bin/env python3
"""
Script to add a new branch to a Hugging Face dataset.
The new branch will be an exact copy of the source branch (default: main).
"""

import argparse
from huggingface_hub import HfApi, create_branch
from huggingface_hub.utils import HfHubHTTPError
import sys


def add_branch_to_dataset(
    dataset_repo: str,
    new_branch: str,
    source_branch: str = "main",
    token: str = None
):
    """
    Create a new branch in a Hugging Face dataset based on an existing branch.

    Args:
        dataset_repo: Dataset repository name (e.g. "ICRA-Competitions/LejuRobotTask1")
        new_branch: Name of the new branch to create (e.g. "v3.0")
        source_branch: Source branch to copy from (default: "main")
        token: Hugging Face authentication token (if not provided, uses the stored token)

    Returns:
        bool: True if the operation was successful, False otherwise
    """
    try:
        # Initialize Hugging Face API
        api = HfApi(token=token)
        
        print(f"Dataset: {dataset_repo}")
        print(f"Creating branch '{new_branch}' from '{source_branch}'...")
        
        # Check that the dataset exists and we have access
        try:
            repo_info = api.dataset_info(dataset_repo)
            print(f"Dataset found")
        except Exception as e:
            print(f"❌ Error accessing dataset: {e}")
            print(f"   Make sure the dataset exists and you have write permissions.")
            return False
        
        # Check if the branch already exists
        try:
            refs = api.list_repo_refs(dataset_repo, repo_type="dataset")
            existing_branches = [branch.name for branch in refs.branches]
            
            if new_branch in existing_branches:
                print(f"Branch '{new_branch}' already exists in the dataset.")
                response = input("Do you want to overwrite it? (y/N): ")
                if response.lower() not in ['y', 'yes', 's', 'si', 'sí']:
                    print("Operation cancelled.")
                    return False
                print(f"Deleting existing branch '{new_branch}'...")
                api.delete_branch(dataset_repo, branch=new_branch, repo_type="dataset")
        except Exception as e:
            print(f"Could not verify existing branches: {e}")
        
        # Create the new branch
        create_branch(
            repo_id=dataset_repo,
            branch=new_branch,
            revision=source_branch,
            repo_type="dataset",
            token=token
        )
        
        print(f"Branch '{new_branch}' created successfully!")
        print(f"You can view it at: https://huggingface.co/datasets/{dataset_repo}/tree/{new_branch}")
        
        return True
        
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"Authentication error: {e}")
            print(f"   Make sure that:")
            print(f"   1. You are logged in: huggingface-cli login")
            print(f"   2. You have write permissions on the dataset")
            print(f"   3. Your token has 'write' scope")
        elif "404" in str(e):
            print(f"Dataset not found: {e}")
            print(f"   Verify that the dataset name is correct: {dataset_repo}")
        else:
            print(f"HTTP error: {e}")
        return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add a new branch to a Hugging Face dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Create branch v3.0 from main
  python add_branch_to_dataset.py ICRA-Competitions/LejuRobotTask1 v3.0
  
  # Create branch v3.0 from another specific branch
  python add_branch_to_dataset.py ICRA-Competitions/LejuRobotTask1 v3.0 --source-branch v2.0
  
  # Use a specific token
  python add_branch_to_dataset.py ICRA-Competitions/LejuRobotTask1 v3.0 --token hf_xxxxx

Note: Make sure you are logged in with: huggingface-cli login
        """
    )
    
    parser.add_argument(
        "dataset_repo",
        help="Dataset repository name (e.g. ICRA-Competitions/LejuRobotTask1)"
    )
    
    parser.add_argument(
        "new_branch",
        help="Name of the new branch to create (e.g. v3.0)"
    )
    
    parser.add_argument(
        "--source-branch",
        default="main",
        help="Source branch to copy from (default: main)"
    )
    
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face authentication token (optional if you are already logged in)"
    )
    
    args = parser.parse_args()
    
    # Run the function
    success = add_branch_to_dataset(
        dataset_repo=args.dataset_repo,
        new_branch=args.new_branch,
        source_branch=args.source_branch,
        token=args.token
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
