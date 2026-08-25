import torch
from torch.utils.data import Dataset
import h5py
import math, random

CHUNK_SIZE = 32

class TrajectoryDataset(Dataset):
    def __init__(self, data_file: str, num_frames: int = 16, embeds_file: str | None = None, indices: list[int] | None = None, deterministic: bool = False):
        self.data_file = data_file
        with h5py.File(self.data_file, 'r') as f:
            self.keys = list(f.keys())
        self.num_frames = num_frames
        self.embeds = None if embeds_file is None else torch.load(embeds_file, map_location='cpu', weights_only=False, mmap=True)
        self.h5_file = None
        self.deterministic = deterministic
            
        self.indices = indices if indices is not None else list(range(len(self.keys)))

    def train(self):
        self.deterministic = False
        return self
        
    def eval(self):
        self.deterministic = True
        return self

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_file, 'r', swmr=True, libver='latest')
            
        index = self.indices[index]
        grp = self.h5_file[self.keys[index]]
        T = grp.attrs['T'] if 'T' in grp.attrs else grp['coords'].shape[0]
        N = grp.attrs['N'] if 'N' in grp.attrs else grp['residues'].shape[0]
        
        gen = torch.Generator().manual_seed(index) if self.deterministic else None
        indices = torch.randint(T, (self.num_frames,), generator=gen) if T < self.num_frames else torch.randperm(T, generator=gen)[:self.num_frames]
        indices, _ = indices.sort()
        idx_list = indices.tolist()
        
        unique_idx, inv = torch.unique(indices, return_inverse=True)
        unique_list = unique_idx.tolist()
        
        coords = torch.from_numpy(grp['coords'][unique_list]).float()[inv]
        if 'energies' in grp:
            interactions = torch.from_numpy(grp['energies'][unique_list]).float()[inv]
        else:
            interactions = torch.zeros((len(idx_list), N, N, 4), dtype=torch.float32)

        return {
            'coords': coords,
            'residues': torch.from_numpy(grp['residues'][:]),
            'props': torch.from_numpy(grp['props'][:]),
            'core_mask': torch.from_numpy(grp['core_mask'][:]),
            'ligand_mask': torch.from_numpy(grp['chain_ids'][:]),
            'interaction_targets': interactions,
            'affinity_target': grp.attrs['affinity'],
            'index': index,
            **({k: v[index] for k, v in self.embeds.items()} if self.embeds is not None else {})
        }

    def split(self, fractions: list[float], seed: int = 42) -> list['TrajectoryDataset']:
        assert math.isclose(sum(fractions), 1.0, abs_tol=1e-5), "Fractions must sum to 1.0"
        
        indices = random.Random(seed).sample(self.indices, len(self.indices))
        sizes = [int(f * len(indices)) for f in fractions[:-1]]
        sizes.append(len(indices) - sum(sizes))
        
        splits, start = [], 0
        for s in sizes:
            dataset = TrajectoryDataset(self.data_file, self.num_frames, indices=indices[start:start+s], deterministic=self.deterministic)
            dataset.embeds = self.embeds
            splits.append(dataset)
            start += s
            
        return splits

    @staticmethod
    def collate_fn(datalist: list[dict]) -> dict[str, torch.Tensor]:
        B = len(datalist)
        N = ((max(len(data['residues']) for data in datalist) - 1) // CHUNK_SIZE + 1) * CHUNK_SIZE
        T = datalist[0]['coords'].shape[0]
        
        batch = {
            'coords': torch.zeros(B, T, N, 4, 3, dtype=torch.float32),
            'chain_ids': torch.zeros(B, N, dtype=torch.int8),
            'residues': torch.zeros(B, N, dtype=torch.int8),
            'props': torch.zeros(B, N, 5, dtype=torch.float32),
            'core_masks': torch.zeros(B, N, dtype=torch.bool),
            'ligand_masks': torch.zeros(B, N, dtype=torch.bool),
            'valid_masks': torch.zeros(B, N, dtype=torch.bool),
            'interaction_targets': torch.zeros(B, T, N, N, 4, dtype=torch.float32),
            'affinity_targets': torch.tensor([i['affinity_target'] for i in datalist], dtype=torch.float32),
            'indices': torch.tensor([i['index'] for i in datalist], dtype=torch.long)
        }
        
        if 'enthalpy_embed_targets' in datalist[0]:
            D = datalist[0]['enthalpy_embed_targets'].shape[-1]
            batch.update({
                'entropy_embed_targets': torch.zeros(B, N, D, dtype=torch.float32),
                'enthalpy_embed_targets': torch.zeros(B, N, N, D, dtype=torch.float32),
            })

        for i, data in enumerate(datalist):
            n = data['residues'].shape[0]
            batch['coords'][i, :, :n] = data['coords']
            batch['residues'][i, :n] = data['residues']
            batch['props'][i, :n] = data['props']
            batch['core_masks'][i, :n] = data['core_mask']
            batch['ligand_masks'][i, :n] = data['ligand_mask']
            batch['interaction_targets'][i, :, :n, :n] = data['interaction_targets']
            batch['valid_masks'][i, :n] = True
            
            if 'enthalpy_embed_targets' in data:
                batch['entropy_embed_targets'][i, :n] = data['entropy_embed_targets']
                batch['enthalpy_embed_targets'][i, :n, :n] = data['enthalpy_embed_targets']
                
        return batch

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = TrajectoryDataset('data/dynarepo/dataset.h5', num_frames=16)
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=TrajectoryDataset.collate_fn, 
        batch_size=8, 
        shuffle=True,
        num_workers=4,
        # persistent_workers=True
    )

    x, y, z, n = 0, 0, 0, 0
    for batch in tqdm(dataloader): 
        x += batch['masks'].sum()
        y += (batch['masks'].unsqueeze(2) & batch['masks'].unsqueeze(1)).sum() * 16
        z += batch['interaction_targets'].numel() / 4
        n += len(batch['masks'])
    print(x / n)
    print(y / n)
    print(z / n)
    print("AA")
