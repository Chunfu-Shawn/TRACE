import math
import random
import torch
from torch.utils.data import Sampler
from collections import Counter

class DistributedBucketSampler(Sampler):
    """ Bucketing by length, then distributed sampling.
    
    Args:
        lengths (List[int]): the list of length of each sample
        batch_size (int): batch size for each progress
        num_replicas (int): world_size
        rank (int): local rank
        shuffle (bool): whether to shuffle samples in each bucket
        drop_last (bool): whether to drop out the last samples less than batch_size
        seed (int): random seed
        cell_types (List[str]): List of cell type strings corresponding to each sample, used for balanced sampling.
        balance_classes (bool): Whether to enable class balancing.
        min_sampling_target (int): The target number of samples to extract per class. 
                                   Classes larger than this will be downsampled. 
                                   Classes smaller than this will be fully sampled (or upsampled if upsample_minority=True).
        upsample_minority (bool): If True, classes with fewer samples than min_sampling_target will be repeatedly sampled 
                                  to strictly match the target. If False, all available samples from the minority class are taken.
    """
    def __init__(self,
                 lengths,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 drop_last=False,
                 seed=0,
                 cell_types=None,
                 balance_classes=False,
                 min_sampling_target=6000, 
                 upsample_minority=False):   
        
        if num_replicas is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Need 'num_replicas' or initialize torch.distributed")
            num_replicas = torch.distributed.get_world_size()

        if rank is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Need 'rank' or initialize torch.distributed")
            rank = torch.distributed.get_rank()

        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # ---------------------------------------------------------
        # [MODIFIED] Balanced sampling initialization logic
        # ---------------------------------------------------------
        self.balance_classes = balance_classes
        self.min_sampling_target = min_sampling_target
        self.upsample_minority = upsample_minority
        
        if self.balance_classes:
            if cell_types is None:
                raise ValueError("cell_types must be provided when balance_classes=True")
            
            # Count occurrences of each cell type string
            counts = Counter(cell_types)
            self.classes = list(counts.keys())
            
            # Group indices by class for dynamic sampling in __iter__
            self.class_to_indices = {c: [] for c in self.classes}
            for idx, c in enumerate(cell_types):
                self.class_to_indices[c].append(idx)
            
            # Calculate the theoretical total samples after balancing
            self.num_samples = 0
            print("[Sampler] Threshold-Capped Balanced Sampling ON:")
            for c in self.classes:
                actual_count = len(self.class_to_indices[c])
                if actual_count >= self.min_sampling_target:
                    self.num_samples += self.min_sampling_target
                    print(f"  - Class '{c}': {actual_count} -> Downsampled to {self.min_sampling_target}")
                else:
                    if self.upsample_minority:
                        self.num_samples += self.min_sampling_target
                        print(f"  - Class '{c}': {actual_count} -> Upsampled to {self.min_sampling_target}")
                    else:
                        self.num_samples += actual_count
                        print(f"  - Class '{c}': {actual_count} -> Fully sampled (Below target threshold)")
                        
            print(f"  -> Total effective samples per Epoch: {self.num_samples}")
        else:
            self.num_samples = len(self.lengths)

        # Strictly calculate the number of batches assigned to the current Rank
        bucket_size = self.batch_size * self.num_replicas
        if self.drop_last:
            self.num_batches_per_replica = math.floor(self.num_samples / bucket_size)
        else:
            self.num_batches_per_replica = math.ceil(self.num_samples / bucket_size)
            
        # Calculate the total required samples globally for subsequent padding to prevent DDP deadlocks
        self.total_size = self.num_batches_per_replica * bucket_size

    def set_epoch(self, epoch: int):
        """ control and confirm the consistence of shuflle for each progress in DDP """
        self.epoch = epoch

    def __iter__(self):
        # set random seed (needed if shuffle or balanced sampling is enabled)
        if self.shuffle or self.balance_classes:
            shuffler = random.Random(self.seed + self.epoch)
            
        # ---------------------------------------------------------
        # 1. Obtain sample indices participating in the training for this Epoch
        # ---------------------------------------------------------
        if self.balance_classes:
            selected_indices = []
            for c in self.classes:
                class_idx_list = self.class_to_indices[c].copy()
                actual_count = len(class_idx_list)
                
                # Shuffle intra-class indices using the epoch-aware shuffler
                shuffler.shuffle(class_idx_list) 
                
                if actual_count >= self.min_sampling_target:
                    # Majority Class: Downsample by truncating
                    selected_indices.extend(class_idx_list[:self.min_sampling_target])
                else:
                    # Minority Class: Below target threshold
                    if self.upsample_minority:
                        # Upsampling: Repeat the shuffled list until it reaches the target
                        full_repeats = self.min_sampling_target // actual_count
                        remainder = self.min_sampling_target % actual_count
                        upsampled_list = class_idx_list * full_repeats + class_idx_list[:remainder]
                        # Reshuffle the repeated list to prevent identical contiguous samples
                        shuffler.shuffle(upsampled_list)
                        selected_indices.extend(upsampled_list)
                    else:
                        # Standard full sampling (No repeats)
                        selected_indices.extend(class_idx_list)

            # global shuffle to mix classes
            if self.shuffle:
                shuffler.shuffle(selected_indices)
        else:
            selected_indices = list(range(len(self.lengths)))

            # global shuffle
            if self.shuffle:
                shuffler.shuffle(selected_indices)

        # ---------------------------------------------------------
        # 2. Padding / Truncating to ensure strict synchronization across DDP processes
        # ---------------------------------------------------------
        if len(selected_indices) < self.total_size:
            padding_size = self.total_size - len(selected_indices)
            selected_indices += selected_indices[:padding_size]
        elif len(selected_indices) > self.total_size:
            selected_indices = selected_indices[:self.total_size]

        # ---------------------------------------------------------
        # 3. Bucket logic applied to the filtered selected_indices
        # ---------------------------------------------------------
        selected_indices.sort(key=lambda idx: self.lengths[idx])

        bucket_size = self.batch_size * self.num_replicas
        buckets = [
            selected_indices[i: i + bucket_size]
            for i in range(0, len(selected_indices), bucket_size)
        ]
        
        if self.shuffle:
            for b in buckets:
                shuffler.shuffle(b)

        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        selected_batches = [
            batch for i, batch in enumerate(all_batches)
            if (i % self.num_replicas) == self.rank
        ]

        if self.shuffle:
            shuffler.shuffle(selected_batches)

        return iter(selected_batches)

    def __len__(self):
        return self.num_batches_per_replica