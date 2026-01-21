import os
from typing import List
from pathlib import Path
import random
from typing import Tuple
import pandas as pd

def get_valid_episodes(repo_id: str) -> List[int]:
    """
    Collects valid episode indices under the lerobot cache for the given repo_id.
    Reads all .parquet files and extracts unique episode_index values.

    Args:
        repo_id (str): HuggingFace repo ID, e.g., 'Qianzhong-Chen/yam_pick_up_cube_sim_rotate_0704'

    Returns:
        List[int]: Sorted list of valid episode indices (e.g., [0, 1, 5, 7, ...])
    """
    
    # Base path to the data directory
    base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id / "data"
    
    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")
    
    # Set to store unique episode indices
    valid_episodes_set = set()

    # Iterate through all chunk directories
    for chunk_dir in base_path.glob("chunk-*"):
        if not chunk_dir.is_dir():
            continue
        
        # Iterate through all .parquet files in the chunk directory
        for parquet_file in chunk_dir.glob("*.parquet"):
            try:
                # Read the parquet file
                df = pd.read_parquet(parquet_file)
                
                # Extract unique episode_index values
                if 'episode_index' in df.columns:
                    episode_indices = df['episode_index'].unique()
                    valid_episodes_set.update(episode_indices)
                else:
                    print(f"Warning: 'episode_index' column not found in {parquet_file}")
            except Exception as e:
                print(f"Error reading {parquet_file}: {e}")

    # Convert set to sorted list
    valid_episodes = sorted(list(valid_episodes_set))
    
    return valid_episodes

def split_train_eval_episodes(valid_episodes: List[int], split_ratio: float = 0.9, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly split valid episodes into training and evaluation sets.

    Args:
        valid_episodes (List[int]): List of valid episode indices.
        split_ratio (float): Fraction of episodes to use for training (default: 0.9).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[List[int], List[int]]: (train_episodes, eval_episodes)
    """
    random.seed(seed)
    episodes = valid_episodes.copy()
    random.shuffle(episodes)

    split_index = int(len(episodes) * split_ratio)
    train_episodes = episodes[:split_index]
    eval_episodes = episodes[split_index:]

    return train_episodes, eval_episodes