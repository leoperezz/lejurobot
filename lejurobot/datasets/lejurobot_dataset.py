import torch
import random
from typing import Dict, List, Tuple
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lejurobot.logger import logger

class LejuRobotDataset(LeRobotDataset):
    """
    Extension of `LeRobotDataset` that correctly handles filtered `episodes`
    when using `delta_timestamps`.

    The original issue is that `LeRobotDataset` assumes that the `idx` received in
    `__getitem__` matches the global frame index (`index`) of the complete dataset.
    This assumption fails when a list of specific `episodes` is provided, because
    `hf_dataset` is compacted (0..N-1), but the metadata (`meta.episodes` and
    fields like `dataset_from_index` / `dataset_to_index`) still reference global indices.

    We solve this by creating a mapping:
        global_index -> local_index_in_hf_dataset

    and by redefining `_get_query_indices` to:
        1. Work in the global index space.
        2. Convert each global index into the corresponding local index of `hf_dataset`.

    This works well in LeRobotDataset V3.0
    """

    def __init__(self, *args, train_with_subtasks: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # Maps to convert between global indices ("index" column)
        # and local indices (position in `hf_dataset`).
        self._global_to_local_index: Dict[int, int] | None = None
        self._local_to_global_index: List[int] | None = None
        self.train_with_subtasks = train_with_subtasks
        
        # Build index maps after initialization if episodes are filtered
        if self.episodes is not None:
            self._build_index_maps()

    def _build_index_maps(self) -> None:
        """
        Builds the maps:
            - global_index (column "index") -> local row index in `hf_dataset`
            - local row index -> global_index

        Only used when `episodes` is not None (subset of the complete dataset).
        """
        if self._global_to_local_index is not None and self._local_to_global_index is not None:
            return

        if self.hf_dataset is None:
            raise RuntimeError("hf_dataset must be loaded before building index maps.")

        self._global_to_local_index = {}
        self._local_to_global_index = []

        # `self.hf_dataset["index"]` returns a list of tensors or ints
        indices_column = self.hf_dataset["index"]
        for local_idx, v in enumerate(indices_column):
            if isinstance(v, torch.Tensor):
                global_idx = int(v.item())
            else:
                global_idx = int(v)
            self._global_to_local_index[global_idx] = local_idx
            self._local_to_global_index.append(global_idx)

    def _get_query_indices(
        self,
        idx: int,
        ep_idx: int,
    ):
        """
        Specialized version to correctly handle subsets of episodes.

        If `episodes` is None, delegates to the original implementation.
        If `episodes` is not None, works with global indices and maps them to
        valid local indices in `hf_dataset` to avoid `IndexError`.
        """
        # Standard case: complete dataset, use original logic as is.
        if self.episodes is None:
            return super()._get_query_indices(idx, ep_idx)

        # If there are no deltas, nothing to do (although in practice this method
        # is called when `delta_indices` is None).
        if self.delta_indices is None:
            return {}, {}

        # Ensure we have the global<->local maps built.
        if self._global_to_local_index is None or self._local_to_global_index is None:
            self._build_index_maps()

        # Episode indices in metadata (always in global space)
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Global index of the current frame (column "index"), not the local idx of hf_dataset
        try:
            current_global_index = self._local_to_global_index[idx]
        except IndexError as e:
            raise IndexError(
                f"Local index {idx} is out of bounds for hf_dataset of size {len(self._local_to_global_index)}"
            ) from e

        query_indices: Dict[str, List[int]] = {}
        padding: Dict[str, torch.BoolTensor] = {}

        for key, delta_idx in self.delta_indices.items():
            global_indices_for_key: List[int] = []
            pad_flags: List[bool] = []

            for delta in delta_idx:
                target_global = current_global_index + delta

                # Clamp to the episode range (in global space, consistent with metadata)
                clamped_target = target_global
                if clamped_target < ep_start:
                    clamped_target = ep_start
                elif clamped_target >= ep_end:
                    clamped_target = ep_end - 1

                # Mark padding if, without clamping, the target would fall outside the episode
                is_pad = (target_global < ep_start) or (target_global >= ep_end)
                pad_flags.append(is_pad)

                # Verify that the global index exists in our filtered dataset
                # This ensures we don't try to access frames that aren't loaded
                if clamped_target not in self._global_to_local_index:
                    raise IndexError(
                        f"Global frame index {clamped_target} (episode {ep_idx}, delta {delta}) "
                        f"not found in hf_dataset subset. Current global index: {current_global_index}, "
                        f"Episode range: [{ep_start}, {ep_end}). "
                        "This usually indicates that some frames from the selected episodes are missing "
                        "in the loaded subset."
                    )

                # Return the global index - _query_hf_dataset will map it to local
                global_indices_for_key.append(clamped_target)

            query_indices[key] = global_indices_for_key
            padding[f"{key}_is_pad"] = torch.BoolTensor(pad_flags)

        return query_indices, padding

    def __getitem__(self, idx) -> dict:
        
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        query_indices = None
        
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name
        
        if self.train_with_subtasks:
            task = self.meta.tasks.iloc[task_idx].name
            next_task = self._get_only_task(idx + 1)
            if task != next_task:
                item["train_with_subtask"] = True
            else:
                if random.random() < 0.3:
                    item["train_with_subtask"] = True
                else:
                    item["train_with_subtask"] = False
        return item

    def _get_only_task(self, idx) -> str:

        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        task_idx = item["task_index"].item()
        task = self.meta.tasks.iloc[task_idx].name
        return task
    
    def get_episode_data_index_for_sampler(self) -> Tuple[List[int], List[int]]:
        """
        Returns adjusted episode indices for use with EpisodeAwareSampler when episodes are filtered.
        
        When specific episodes are selected, the metadata still contains global indices,
        but the actual hf_dataset only contains frames from selected episodes.
        This method returns the correct relative indices that match the filtered dataset.
        
        Returns:
            Tuple of (from_indices, to_indices) adjusted for the filtered dataset.
        """
        if self.episodes is None:
            # No filtering, return original indices
            return (
                list(self.meta.episodes["dataset_from_index"]),
                list(self.meta.episodes["dataset_to_index"])
            )
        
        # Ensure index maps are built
        if self._global_to_local_index is None:
            self._build_index_maps()
        
        from_indices = []
        to_indices = []
        
        # For each selected episode, compute its relative indices in the filtered dataset
        for ep_idx in self.episodes:
            ep = self.meta.episodes[ep_idx]
            global_from = ep["dataset_from_index"]
            global_to = ep["dataset_to_index"]
            
            # Map global indices to local indices
            try:
                local_from = self._global_to_local_index[global_from]
                # dataset_to_index is exclusive, so we need to get the last valid index + 1
                # Find the local index for the last frame in the episode
                local_to = self._global_to_local_index[global_to - 1] + 1
                
                from_indices.append(local_from)
                to_indices.append(local_to)
            except KeyError as e:
                logger.warning(f"Episode {ep_idx} has missing frames in filtered dataset: {e}")
                continue
        
        return from_indices, to_indices