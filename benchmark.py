import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TeacherModel, StudentModel
from dataset import TrajectoryDataset

def benchmark(data_file: str, model_path: str | list[str], device='mps', batch_size=8, num_frames=64, plot=False, save=False):

    if not os.path.exists(data_file):
        print(f"Test dataset '{data_file}' not found.")
        return

    dataset = TrajectoryDataset(data_file, num_frames=num_frames).eval()
    # _, dataset = dataset.split([0.9, 0.1], seed=0xDEAFBEEF)

    dataloader = DataLoader(
        dataset=dataset, 
        collate_fn=TrajectoryDataset.collate_fn, 
        batch_size=batch_size, 
        shuffle=False,
    )

    model_paths = [model_path] if isinstance(model_path, str) else model_path
    for path in model_paths:
        if not os.path.exists(path): raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        
        model = StudentModel().to(device).eval()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {os.path.basename(path)}"):
                batch = {k:x.to(device) for k, x in batch.items()}
                
                affinity, _, _ = model(**batch)
                
                all_preds.append(affinity.cpu())
                all_targets.append(batch['affinity_targets'].cpu())
                
                if device == 'mps': torch.mps.empty_cache()

        preds = torch.concat(all_preds, dim=0).numpy()
        targets = torch.concat(all_targets, dim=0).numpy()
        
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))

        pearson_r, pearson_p = pearsonr(preds, targets)
        spearman_rho, spearman_p = spearmanr(preds, targets)

        print("\n==================================================")
        print("  FINAL TEST SET PERFORMANCE")
        print("==================================================")
        print(f"Model: {path} | Frames: {num_frames}")
        print(f"Total Complexes Evaluated : {len(preds)}")
        print(f"RMSE (Root Mean Sq Error) : {rmse:.3f}")
        print(f"MAE  (Mean Abs Error)     : {mae:.3f}")
        print("--------------------------------------------------")
        print(f"Pearson R (Linear)        : {pearson_r:.3f}  (p={pearson_p:.2e})")
        print(f"Spearman \u03C1 (Ranking)      : {spearman_rho:.3f}  (p={spearman_p:.2e})")
        print("==================================================\n")
        
        if plot:
            plt.figure(figsize=(8, 6))
            plt.scatter(targets, preds, alpha=0.5, edgecolors='k', s=20)
            
            plt.xlabel("Experimental Affinity ($pK_d / pK_i$)")
            plt.ylabel("Model Predicted Score")
            plt.title(f"Ranking Correlation\nSpearman $\\rho$: {spearman_rho:.3f} | Pearson R: {pearson_r:.3f}")
            
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            if save:
                plt.savefig(path[:-3] + '.png', dpi=300)
                plt.close()
            else: plt.show()

if __name__ == '__main__':
    benchmark(
        data_file='data/kastritis/dataset.h5', 
        model_path=[f'checkpoints/run85_84_finetune/epoch_{i}.pt' for i in range(1,50)],
        device='mps',
        batch_size=8,
        num_frames=1,
        plot=True,
        save=True
    )