"""
evaluate.py - Model Checkpoint Evaluation & Verification Script

Evaluates trained StudentModel (or TeacherModel) checkpoints on protein-protein
binding affinity datasets (e.g. Kastritis Affinity Benchmark, PDBbind).
Computes standard regression and ranking metrics (Pearson r, Spearman rho, MAE, RMSE, R²),
tracks peak memory usage, and optionally exports per-complex predictions to CSV.
"""

import os
import sys
import argparse
import platform
import resource
import psutil
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import StudentModel, TeacherModel
from dataset import TrajectoryDataset


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Resolve device string to torch.device with auto-detection fallback.
    Order of preference for 'auto': CUDA -> MPS -> CPU.
    """
    if device_str is None or device_str.lower() == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    device = torch.device(device_str)
    # Validate device availability
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested ({device_str}) but not available. Falling back to CPU.", file=sys.stderr)
        return torch.device('cpu')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        print(f"Warning: MPS requested ({device_str}) but not available. Falling back to CPU.", file=sys.stderr)
        return torch.device('cpu')
    
    return device


def get_peak_memory_gb() -> float:
    """
    Return peak process memory in Gigabytes (GB).
    Handles OS differences in resource.ru_maxrss (bytes on macOS, KB on Linux)
    and queries psutil RSS.
    """
    process = psutil.Process()
    rss_gb = process.memory_info().rss / (1024 ** 3)
    
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':
        rusage_gb = rusage / (1024 ** 3)
    else:
        rusage_gb = (rusage * 1024) / (1024 ** 3)
    
    cuda_gb = 0.0
    if torch.cuda.is_available():
        try:
            cuda_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
            
    return max(rss_gb, rusage_gb, cuda_gb)


def load_model_and_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_type: str = 'auto'
) -> torch.nn.Module:
    """
    Instantiate model and load weights from checkpoint.
    Supports auto-detecting StudentModel vs TeacherModel from state dict keys.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    if model_type == 'auto':
        # TeacherModel has log_beta or interaction_head in state dict
        if 'log_beta' in state_dict or 'interaction_head.weight' in state_dict:
            model = TeacherModel().to(device)
        else:
            model = StudentModel().to(device)
    elif model_type.lower() == 'teacher':
        model = TeacherModel().to(device)
    elif model_type.lower() == 'student':
        model = StudentModel().to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'auto', 'student', or 'teacher'.")

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def evaluate(
    dataset_path: str,
    checkpoint_path: str,
    batch_size: int = 8,
    device: str = 'auto',
    num_frames: int = 1,
    num_workers: int = 0,
    output_csv: str | None = None,
    plot_path: str | None = None,
    quiet: bool = False
) -> dict:
    """
    Run evaluation of a model checkpoint on an HDF5 dataset.

    Args:
        dataset_path: Path to dataset.h5 file.
        checkpoint_path: Path to .pt model checkpoint.
        batch_size: DataLoader batch size.
        device: Device string ('auto', 'cuda', 'mps', 'cpu').
        num_frames: Number of temporal frames (default 1 for static structures).
        num_workers: Number of workers for DataLoader.
        output_csv: Optional file path to export per-complex prediction CSV.
        plot_path: Optional file path to save scatter plot PNG.
        quiet: If True, suppress stdout printing and progress bar.

    Returns:
        Dictionary containing metric summaries, sample counts, and predictions.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    resolved_device = get_device(device)
    model = load_model_and_checkpoint(checkpoint_path, resolved_device)

    dataset = TrajectoryDataset(dataset_path, num_frames=num_frames).eval()
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {dataset_path} contains 0 entries.")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TrajectoryDataset.collate_fn
    )

    all_preds = []
    all_targets = []
    all_enthalpies = []
    all_entropies = []
    all_indices = []
    peak_mem = get_peak_memory_gb()

    iterator = dataloader if quiet else tqdm(dataloader, desc=f"Evaluating {os.path.basename(dataset_path)}")

    with torch.no_grad():
        for batch in iterator:
            # Move tensor inputs to target device
            batch_device = {
                k: v.to(resolved_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**batch_device)
            if isinstance(outputs, tuple):
                affinity = outputs[0]
                enthalpy = outputs[1] if len(outputs) > 1 else torch.zeros_like(affinity)
                entropy = outputs[2] if len(outputs) > 2 else torch.zeros_like(affinity)
            elif isinstance(outputs, dict):
                affinity = outputs.get('affinities', outputs.get('affinity'))
                enthalpy = outputs.get('enthalpy', torch.zeros_like(affinity))
                entropy = outputs.get('entropy', torch.zeros_like(affinity))
            else:
                affinity = outputs
                enthalpy = torch.zeros_like(affinity)
                entropy = torch.zeros_like(affinity)

            all_preds.append(affinity.detach().cpu())
            all_targets.append(batch['affinity_targets'].detach().cpu())
            all_enthalpies.append(enthalpy.detach().cpu() if isinstance(enthalpy, torch.Tensor) else torch.tensor(enthalpy))
            all_entropies.append(entropy.detach().cpu() if isinstance(entropy, torch.Tensor) else torch.tensor(entropy))
            all_indices.extend(batch['indices'].tolist())

            if resolved_device.type == 'mps':
                torch.mps.empty_cache()
            elif resolved_device.type == 'cuda':
                torch.cuda.empty_cache()

            peak_mem = max(peak_mem, get_peak_memory_gb())

    preds = torch.concat(all_preds, dim=0).numpy().astype(np.float64)
    targets = torch.concat(all_targets, dim=0).numpy().astype(np.float64)
    enthalpies = torch.concat(all_enthalpies, dim=0).numpy().astype(np.float64)
    entropies = torch.concat(all_entropies, dim=0).numpy().astype(np.float64)

    # Sanity checks
    if np.isnan(preds).any() or np.isinf(preds).any():
        raise RuntimeError("Model inference produced NaN or Inf predictions.")
    if np.isnan(targets).any() or np.isinf(targets).any():
        raise RuntimeError("Target affinities contain NaN or Inf values.")

    # Match complex IDs from dataset
    complex_ids = [dataset.keys[idx] for idx in all_indices]

    # Calculate metrics
    errors = preds - targets
    abs_errors = np.abs(errors)
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_errors))
    std_err = float(np.std(errors))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

    if len(preds) > 1 and np.std(preds) > 1e-9 and np.std(targets) > 1e-9:
        p_res = pearsonr(preds, targets)
        pearson_r, pearson_p = float(p_res.statistic), float(p_res.pvalue)
        s_res = spearmanr(preds, targets)
        spearman_rho, spearman_p = float(s_res.statistic), float(s_res.pvalue)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0

    peak_mem = max(peak_mem, get_peak_memory_gb())

    # Build results dictionary
    results = {
        'num_samples': len(preds),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'std_err': std_err,
        'peak_memory_gb': peak_mem,
        'device': str(resolved_device),
        'dataset_path': dataset_path,
        'checkpoint_path': checkpoint_path,
        'complex_ids': complex_ids,
        'targets': targets,
        'predictions': preds,
        'enthalpies': enthalpies,
        'entropies': entropies
    }

    # Save to CSV if requested
    if output_csv:
        csv_dir = os.path.dirname(os.path.abspath(output_csv))
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        df = pd.DataFrame({
            'complex_id': complex_ids,
            'target_affinity': np.round(targets, 4),
            'predicted_affinity': np.round(preds, 4),
            'predicted_enthalpy': np.round(enthalpies, 4),
            'predicted_entropy': np.round(entropies, 4),
            'error': np.round(errors, 4),
            'absolute_error': np.round(abs_errors, 4)
        })
        df.to_csv(output_csv, index=False)
        if not quiet:
            print(f"Predictions saved to CSV: {output_csv}")

    # Generate scatter plot if requested
    if plot_path:
        plot_dir = os.path.dirname(os.path.abspath(plot_path))
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        plt.scatter(targets, preds, color='#1f77b4', alpha=0.7, edgecolors='k', s=35, label='Complexes')
        
        # Identity line
        min_val = min(targets.min(), preds.min()) - 0.5
        max_val = max(targets.max(), preds.max()) + 0.5
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Ideal ($y=x$)')
        
        # Best fit line
        if len(targets) > 1:
            slope, intercept = np.polyfit(targets, preds, 1)
            x_vals = np.linspace(min_val, max_val, 100)
            plt.plot(x_vals, slope * x_vals + intercept, 'b-', alpha=0.7, label=f'Fit (slope={slope:.2f})')

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.xlabel("Experimental Affinity ($pK_d$)", fontsize=11)
        plt.ylabel("Model Predicted Affinity ($pK_d$)", fontsize=11)
        plt.title(
            f"Model Evaluation: {os.path.basename(dataset_path)}\n"
            f"Pearson $r = {pearson_r:.3f}$ | Spearman $\\rho = {spearman_rho:.3f}$ | RMSE = ${rmse:.3f}$",
            fontsize=11
        )
        plt.legend(loc='upper left', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        if not quiet:
            print(f"Evaluation plot saved to: {plot_path}")

    # Display report if not quiet
    if not quiet:
        print_evaluation_report(results)

    return results


def print_evaluation_report(results: dict):
    """Print formatted evaluation summary table."""
    mem_status = "PASSED" if results['peak_memory_gb'] < 40.0 else "EXCEEDED"
    print("\n" + "=" * 68)
    print("                 TRAJ-BIND MODEL EVALUATION REPORT")
    print("=" * 68)
    print(f"  Dataset Path       : {results['dataset_path']}")
    print(f"  Checkpoint Path    : {results['checkpoint_path']}")
    print(f"  Compute Device     : {results['device']}")
    print(f"  Total Complexes    : {results['num_samples']}")
    print("-" * 68)
    print("  PERFORMANCE METRICS:")
    print(f"  Pearson Correlation (r)       : {results['pearson_r']:>8.4f}  (p = {results['pearson_p']:.2e})")
    print(f"  Spearman Correlation (ρ)      : {results['spearman_rho']:>8.4f}  (p = {results['spearman_p']:.2e})")
    print(f"  Mean Absolute Error (MAE)     : {results['mae']:>8.4f}")
    print(f"  Root Mean Squared Error (RMSE): {results['rmse']:>8.4f}")
    print(f"  Mean Squared Error (MSE)      : {results['mse']:>8.4f}")
    print(f"  Coefficient of Determ. (R²)   : {results['r2']:>8.4f}")
    print(f"  Std Error of Residuals        : {results['std_err']:>8.4f}")
    print("-" * 68)
    print("  RESOURCE CONSTRAINTS:")
    print(f"  Peak Memory Consumption       : {results['peak_memory_gb']:>8.2f} GB (<40.0 GB: {mem_status})")
    print("=" * 68 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Traj-Bind student model checkpoint on protein affinity datasets.")
    parser.add_argument('--dataset_path', type=str, default='data/kastritis/dataset.h5',
                        help="Path to HDF5 dataset file (default: data/kastritis/dataset.h5)")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/run84_82_student/epoch_85.pt',
                        help="Path to model checkpoint (default: checkpoints/run84_82_student/epoch_85.pt)")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Inference batch size (default: 8)")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument('--num_frames', type=int, default=1,
                        help="Number of frames per complex (default: 1 for static structures)")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of DataLoader workers (default: 0)")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="Path to save predictions as CSV (optional)")
    parser.add_argument('--plot', type=str, default=None,
                        help="Path to save scatter plot PNG (optional)")
    parser.add_argument('--quiet', action='store_true',
                        help="Suppress verbose table printing")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        output_csv=args.output_csv,
        plot_path=args.plot,
        quiet=args.quiet
    )
