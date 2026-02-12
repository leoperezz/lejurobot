#!/usr/bin/env python3

"""
Script to download the complete lerobot/aloha_mobile_wash_pan dataset repository from Hugging Face
"""

import os
import subprocess
from pathlib import Path


def download_dataset():
    """Download the complete Hugging Face dataset repository"""

    # Configuration
    dataset_id = "ICRA-Competitions/LejuRobotTask1"
    branch = "main"  # Specify the branch to download

    output_dir = "/home/lperez/main/NONHUMAN/lejurobot/outputs/dataset/LejuRobotTask1"
    full_output_path = output_dir

    # Create output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading complete dataset repository: {dataset_id}")
    print(f"Branch: {branch}")
    print(f"Output directory: {full_output_path}")

    try:
        # Use huggingface_hub to download the entire repository
        from huggingface_hub import snapshot_download

        # Download the entire repository from specific branch
        repo_path = snapshot_download(
            repo_id=dataset_id,
            local_dir=full_output_path,
            repo_type="dataset",
            revision=branch
        )

        print(f"✅ Complete dataset repository downloaded to: {repo_path}")
        print(f"📁 Repository contents:")

        # List contents of the downloaded repository
        for root, dirs, files in os.walk(repo_path):
            level = root.replace(repo_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Show first 10 files
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")

    except Exception as e:
        print(f"❌ Error downloading the dataset repository: {e}")
        return False

    return True


if __name__ == "__main__":
    download_dataset()