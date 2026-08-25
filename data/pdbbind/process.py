#!/usr/bin/env python3
"""
PDBbind Protein-Protein Dataset Processing Pipeline
Adapted from data/dynarepo/process.py and data/kastritis/process.py for static crystal structures (T=1).

Ingests the filtered clean non-homologous PDBbind complexes (from data/pdbbind/index.csv)
and raw PDB crystal structures (from data/pdbbind/raw/*.pdb).
Extracts 3D coordinates (CA, C, N, sidechain centroids), residue types, chemical properties,
chain partner indicators, and core interface masks.
Generates intermediate .pt.gz files in data/pdbbind/dataset/ and combines them into data/pdbbind/dataset.h5.
Appends processing documentation and metrics to data/pdbbind/info.txt.
"""

import os
import sys
import io
import csv
import gzip
import argparse
from datetime import datetime, timezone
from glob import glob
import numpy as np
import pandas as pd
import torch
import h5py
import MDAnalysis as mda
from tqdm import tqdm
import concurrent.futures

# Residue ID mapping (1..29) conforming strictly to dynarepo schema
RES_MAP = {
    'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 'GLU': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10,
    'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20,
    'HSD': 21, 'HSE': 22, 'HSP': 23, 'ASH': 24, 'GLH': 25, 'LYN': 26, 'TYM': 27, 'CYX': 28, 'CYM': 29 
}

RES_ALIASES = {
    'HID': 'HSD', 'HIE': 'HSE', 'HIP': 'HSP', 'AR0': 'ARG', 'MSE': 'MET',
    'SEC': 'CYS', 'CYX': 'CYX', 'CYM': 'CYM'
}

# 5-dimensional physicochemical properties:
# [charge, hydropathy, hbond_donors, hbond_acceptors, volume / 100]
RES_PROPERTIES = {
    'ALA': [ 0.0,  1.8, 0, 0,  88.6 / 100], 'ARG': [ 1.0, -4.5, 3, 0, 173.4 / 100],
    'ASN': [ 0.0, -3.5, 1, 1, 114.1 / 100], 'ASP': [-1.0, -3.5, 0, 2, 111.1 / 100],
    'CYS': [ 0.0,  2.5, 0, 0, 108.5 / 100], 'GLN': [ 0.0, -3.5, 1, 1, 143.8 / 100],
    'GLU': [-1.0, -3.5, 0, 2, 138.4 / 100], 'GLY': [ 0.0, -0.4, 0, 0,  60.1 / 100],
    'HIS': [ 0.5, -3.2, 1, 1, 153.2 / 100], 'ILE': [ 0.0,  4.5, 0, 0, 166.7 / 100],
    'LEU': [ 0.0,  3.8, 0, 0, 166.7 / 100], 'LYS': [ 1.0, -3.9, 1, 0, 168.6 / 100],
    'MET': [ 0.0,  1.9, 0, 0, 162.9 / 100], 'PHE': [ 0.0,  2.8, 0, 0, 189.9 / 100],
    'PRO': [ 0.0, -1.6, 0, 0, 112.7 / 100], 'SER': [ 0.0, -0.8, 1, 1,  89.0 / 100],
    'THR': [ 0.0, -0.7, 1, 1, 116.1 / 100], 'TRP': [ 0.0, -0.9, 1, 0, 227.8 / 100],
    'TYR': [ 0.0, -1.3, 1, 1, 193.6 / 100], 'VAL': [ 0.0,  4.2, 0, 0, 140.0 / 100],
    'HSD': [ 0.0, -3.2, 1, 1, 153.2 / 100], 'HSE': [ 0.0, -3.2, 1, 1, 153.2 / 100],
    'HSP': [ 1.0, -3.2, 2, 0, 153.2 / 100], 'ASH': [ 0.0, -3.5, 1, 1, 111.1 / 100],
    'GLH': [ 0.0, -3.5, 1, 1, 138.4 / 100], 'LYN': [ 0.0, -3.9, 1, 1, 168.6 / 100],
    'TYM': [-1.0, -1.3, 0, 2, 193.6 / 100], 'CYX': [ 0.0,  2.5, 0, 0, 108.5 / 100],
    'CYM': [-1.0,  2.5, 0, 0, 108.5 / 100]
}


def init_worker():
    """Initializes worker processes with muted PyRosetta context if available."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    try:
        import pyrosetta
        pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false -load_PDB_components false")
    except Exception:
        pass


def load_index(index_path: str = "data/pdbbind/index.csv") -> list:
    """Loads parsed PDBbind filtered index records from CSV."""
    records = []
    with open(index_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row['pdb_id'].strip().lower()
            records.append({
                'complex_id': pdb_id,
                'pdb_id': pdb_id,
                'pdb_file': f"{pdb_id}.pdb",
                'affinity': float(row['affinity']),
                'receptor': set(row['receptor'].strip()),
                'ligand': set(row['ligand'].strip()),
                'resolution': row.get('resolution', 'N/A'),
                'year': row.get('year', 'N/A'),
                'max_bench_sim': float(row.get('max_bench_sim', 0.0)),
                'best_bench_hit': row.get('best_bench_hit', 'none')
            })
    return records


def get_residue_chain(res) -> str:
    """Extracts chain / segment identifier from an MDAnalysis residue."""
    segid = getattr(res, 'segid', None)
    if segid and segid.strip():
        return segid.strip()
    chainID = getattr(res, 'chainID', None)
    if chainID and chainID.strip():
        return chainID.strip()
    if hasattr(res.atoms, 'segids') and len(res.atoms.segids) > 0 and res.atoms.segids[0].strip():
        return res.atoms.segids[0].strip()
    if hasattr(res.atoms, 'chainIDs') and len(res.atoms.chainIDs) > 0 and res.atoms.chainIDs[0].strip():
        return res.atoms.chainIDs[0].strip()
    return ''


def process_one(pdb_path: str, metadata: dict, processed_dir: str) -> tuple:
    """
    Processes a single static crystal structure (T=1).
    Extracts coordinates, residues, properties, chain IDs, and core mask.
    Saves intermediate .pt.gz file.
    
    Returns:
        (success: bool, info: dict, warnings: list[str])
    """
    warnings = []
    complex_id = metadata['complex_id']
    try:
        if not os.path.exists(pdb_path):
            return False, {}, [f"  [❌] [Error] {complex_id}: PDB file not found at {pdb_path}"]

        # Load static PDB with MDAnalysis (Model 1)
        u = mda.Universe(pdb_path)
        if len(u.trajectory) > 0:
            u.trajectory[0]
        pos = u.atoms.positions

        md_res = []
        for res in u.select_atoms("protein").residues:
            name = RES_ALIASES.get(res.resname.upper(), res.resname.upper())
            if name not in RES_MAP:
                continue

            atoms = {a.name: a.index for a in res.atoms}
            if not {'N', 'CA', 'C'}.issubset(atoms.keys()):
                warnings.append(f"  [⚠] [Warning] {complex_id}: Dropping {name} {res.resnum} missing backbone.")
                continue

            sc = [a.index for a in res.atoms if a.name not in {'C', 'CA', 'N', 'O', 'HN', 'HA', 'H'}]
            
            chain_id = get_residue_chain(res)
            if chain_id in metadata['receptor']:
                partner = 0
            elif chain_id in metadata['ligand']:
                partner = 1
            else:
                continue

            md_res.append({
                'id': RES_MAP[name],
                'props': RES_PROPERTIES[name],
                'ca': atoms['CA'],
                'c': atoms['C'],
                'n': atoms['N'],
                'sc': sc if sc else [atoms['CA']],
                'chain': partner,
                'name': name,
                'resnum': res.resnum,
                'chain_id': chain_id
            })

        if not md_res:
            return False, {}, warnings + [f"  [❌] [Skip] {complex_id}: No valid protein residues found."]

        chains = np.array([r['chain'] for r in md_res])
        if 0 not in chains or 1 not in chains:
            return False, {}, warnings + [f"  [❌] [Skip] {complex_id}: Missing binding partner in protein atoms (chains={set(chains)})."]

        # Compute pairwise sidechain centroid cross-partner distances
        sc_coords = np.array([pos[r['sc']].mean(axis=0) for r in md_res])
        sc_dists = np.linalg.norm(sc_coords[:, None, :] - sc_coords[None, :, :], axis=2)
        min_dists = np.where(
            chains == 0,
            sc_dists[:, chains == 1].min(axis=1),
            sc_dists[:, chains == 0].min(axis=1)
        )

        # Distance thresholding: keep interface within 25 Å, core within 15 Å
        keep_mask = min_dists <= 25.0
        core_mask = min_dists[keep_mask] <= 15.0
        valid_res = [md_res[i] for i in range(len(md_res)) if keep_mask[i]]
        N = len(valid_res)

        if N == 0:
            return False, {}, warnings + [f"  [❌] [Skip] {complex_id}: Interface empty after 25Å cutoff."]

        valid_chains = np.array([r['chain'] for r in valid_res])
        if 0 not in valid_chains or 1 not in valid_chains:
            return False, {}, warnings + [f"  [❌] [Skip] {complex_id}: Interface contains only one partner."]

        ca_idx = [r['ca'] for r in valid_res]
        c_idx = [r['c'] for r in valid_res]
        n_idx = [r['n'] for r in valid_res]

        # 4-coordinate representation per residue: CA (0), C (1), N (2), SC centroid (3)
        frame_c = np.zeros((N, 4, 3), dtype=np.float32)
        frame_c[:, 0] = pos[ca_idx]
        frame_c[:, 1] = pos[c_idx]
        frame_c[:, 2] = pos[n_idx]
        for i, r in enumerate(valid_res):
            frame_c[i, 3] = pos[r['sc']].mean(axis=0)

        # Single frame evaluation with T=1 leading dimension: [1, N, 4, 3]
        coords_1frame = np.expand_dims(frame_c, axis=0).astype(np.float16)

        out_data = {
            'T': 1,
            'N': N,
            'dynarepo_id': complex_id,
            'complex_id': complex_id,
            'affinity': float(metadata['affinity']),
            'coords': torch.from_numpy(coords_1frame),
            'residues': torch.from_numpy(np.array([r['id'] for r in valid_res], dtype=np.int8)),
            'props': torch.from_numpy(np.array([r['props'] for r in valid_res], dtype=np.float32)),
            'chain_ids': torch.from_numpy(np.array([r['chain'] for r in valid_res], dtype=np.int8)),
            'core_mask': torch.from_numpy(core_mask)
        }

        os.makedirs(processed_dir, exist_ok=True)
        pt_path = os.path.join(processed_dir, f"{complex_id}.pt.gz")
        with gzip.open(pt_path, 'wb') as f:
            torch.save(out_data, f)

        stat_info = {
            'complex_id': complex_id,
            'pdb_id': metadata['pdb_id'],
            'N': N,
            'core_N': int(core_mask.sum()),
            'rec_N': int((valid_chains == 0).sum()),
            'lig_N': int((valid_chains == 1).sum()),
            'affinity': float(metadata['affinity']),
            'resolution': metadata.get('resolution', 'N/A'),
            'year': metadata.get('year', 'N/A'),
            'max_bench_sim': float(metadata.get('max_bench_sim', 0.0)),
            'best_bench_hit': metadata.get('best_bench_hit', 'none'),
            'warnings': warnings
        }
        return True, stat_info, warnings + [f"  [✔] Processed {complex_id} (T=1, N={N}, core={int(core_mask.sum())})"]

    except Exception as e:
        return False, {}, warnings + [f"  [❌] [Error] Failed on {complex_id}: {e}"]


def process_all(
    raw_dir: str = "data/pdbbind/raw",
    processed_dir: str = "data/pdbbind/dataset",
    index_path: str = "data/pdbbind/index.csv",
    info_path: str = "data/pdbbind/info.txt"
) -> list:
    """
    Processes all filtered PDBbind complexes in parallel, extracts intermediate .pt.gz files,
    and appends comprehensive info.txt log.
    """
    index_records = load_index(index_path)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(info_path)), exist_ok=True)

    tasks = []
    for info in index_records:
        pdb_file = info['pdb_file']
        pdb_path = os.path.join(raw_dir, pdb_file)
        if not os.path.exists(pdb_path):
            pdb_alt = os.path.join(raw_dir, f"{info['pdb_id'].upper()}.pdb")
            if os.path.exists(pdb_alt):
                pdb_path = pdb_alt
            else:
                pdb_alt_lower = os.path.join(raw_dir, f"{info['pdb_id'].lower()}.pdb")
                if os.path.exists(pdb_alt_lower):
                    pdb_path = pdb_alt_lower
        tasks.append((pdb_path, info, processed_dir))

    print(f"\nProcessing {len(tasks)} PDBbind complexes for static tensor extraction (T=1)...")

    max_workers = min(os.cpu_count() or 4, 16)
    successful_stats = []
    failed_logs = []
    all_warnings = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        future_to_task = {executor.submit(process_one, *t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Extracting PDBbind Complexes"):
            task_arg = future_to_task[future]
            try:
                success, stat_info, msgs = future.result()
                for msg in msgs:
                    if '[⚠]' in msg:
                        all_warnings.append(msg.strip())
                    if '[❌]' in msg:
                        failed_logs.append(msg.strip())
                if success:
                    successful_stats.append(stat_info)
            except Exception as exc:
                err_msg = f"  [❌] [Fatal] Worker failed on {task_arg[1]['complex_id']}: {exc}"
                failed_logs.append(err_msg)
                tqdm.write(err_msg)

    # Sort successful statistics by complex_id
    successful_stats.sort(key=lambda x: x['complex_id'])

    # Append processing statistics to info.txt
    append_info_txt(info_path, len(index_records), successful_stats, failed_logs, all_warnings)
    print(f"\nSuccessfully extracted {len(successful_stats)} / {len(tasks)} complexes to {processed_dir}")
    print(f"Processing documentation updated in {info_path}")
    return successful_stats


def combine_to_h5(
    processed_dir: str = "data/pdbbind/dataset",
    h5_path: str = "data/pdbbind/dataset.h5",
    index_path: str = "data/pdbbind/index.csv"
):
    """
    Combines intermediate .pt.gz files into final HDF5 dataset conforming to TrajectoryDataset schema.
    Strictly iterates over approved complexes in index.csv.
    """
    index_records = load_index(index_path)
    pt_files = []
    for r in index_records:
        pt_p = os.path.join(processed_dir, f"{r['complex_id']}.pt.gz")
        if os.path.exists(pt_p):
            pt_files.append(pt_p)

    if not pt_files:
        raise RuntimeError(f"No valid .pt.gz files found in {processed_dir} matching index {index_path}.")

    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    print(f"\nCombining {len(pt_files)} intermediate tensor files into {h5_path} (from {len(index_records)} index records)...")

    with h5py.File(h5_path, 'w') as h5f:
        for pt_file in tqdm(pt_files, desc="Writing HDF5 Groups"):
            with gzip.open(pt_file, 'rb') as f:
                data = torch.load(f, weights_only=False)

            complex_id = data['dynarepo_id']
            N = data['N']

            grp = h5f.create_group(complex_id)
            grp.attrs['T'] = 1
            grp.attrs['N'] = N
            grp.attrs['dynarepo_id'] = complex_id
            grp.attrs['complex_id'] = complex_id
            grp.attrs['affinity'] = float(data['affinity'])

            # Store datasets with gzip compression level 4 for coordinates
            grp.create_dataset(
                'coords',
                data=data['coords'].numpy(),
                chunks=(1, N, 4, 3),
                compression='gzip',
                compression_opts=4
            )
            grp.create_dataset('residues', data=data['residues'].numpy())
            grp.create_dataset('props', data=data['props'].numpy())
            grp.create_dataset('chain_ids', data=data['chain_ids'].numpy())
            grp.create_dataset('core_mask', data=data['core_mask'].numpy())

    print(f"Successfully created static HDF5 dataset at {h5_path} ({len(pt_files)} groups).")


def append_info_txt(
    info_path: str,
    total_requested: int,
    stats: list,
    failures: list,
    warnings: list
):
    """Appends processing documentation and metrics to info.txt."""
    # Read existing content if available to ensure we preserve M2 filter logs
    existing_content = ""
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            existing_content = f.read()

    # If already contains Section 7, slice before Section 7 to refresh
    split_marker = "\n--- 7. STATIC STRUCTURE PROCESSING & HDF5 GENERATION REPORT ---"
    if split_marker in existing_content:
        base_content = existing_content.split(split_marker)[0].rstrip()
    else:
        base_content = existing_content.rstrip()

    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report_lines = [
        "",
        "--- 7. STATIC STRUCTURE PROCESSING & HDF5 GENERATION REPORT ---",
        f"Timestamp                  : {timestamp_str}",
        "Pipeline Step              : Milestone M4 (PDBbind Static Tensor Processing & HDF5 Generation)",
        f"Total Retained Index Inputs: {total_requested}",
        f"Successfully Processed     : {len(stats)} ({len(stats)/total_requested*100:.2f}%)",
        f"Skipped / Incompatible     : {len(failures)} ({len(failures)/total_requested*100:.2f}%)",
        f"Total Warnings Logged      : {len(warnings)}",
        "",
        "--- 8. DATASET REPRESENTATION SPECIFICATIONS ---",
        "Temporal Dimension         : T = 1 (Static crystal structure representation)",
        "Interface Retention Cutoff : Residues with min cross-partner sidechain dist <= 25.0 A",
        "Core Interface Definition  : Interface residues with min cross-partner sidechain dist <= 15.0 A",
        "Coordinate Tensor Shape    : (1, N, 4, 3) float16, gzip level 4 [CA=0, C=1, N=2, Sidechain Centroid=3]",
        "Residue Types Dataset      : (N,) int8 [IDs 1..29 conforming to dynarepo schema]",
        "Chemical Properties Dataset: (N, 5) float32 [Charge, Hydropathy, Donors, Acceptors, Volume/100]",
        "Chain Indicators Dataset   : (N,) int8 [0 = Receptor, 1 = Ligand]",
        "Core Mask Dataset          : (N,) bool [distance <= 15 A]",
        ""
    ]

    if stats:
        n_vals = [s['N'] for s in stats]
        core_vals = [s['core_N'] for s in stats]
        aff_vals = [s['affinity'] for s in stats]

        report_lines.extend([
            "--- 9. INTERFACE RESIDUE AND AFFINITY DISTRIBUTIONS ---",
            "Interface Residue Count (N, <= 25A):",
            f"  Min: {min(n_vals)}, Max: {max(n_vals)}, Mean: {np.mean(n_vals):.2f}, Median: {np.median(n_vals):.1f}, Std: {np.std(n_vals):.2f}",
            "Core Interface Residue Count (dist <= 15A):",
            f"  Min: {min(core_vals)}, Max: {max(core_vals)}, Mean: {np.mean(core_vals):.2f}, Median: {np.median(core_vals):.1f}, Std: {np.std(core_vals):.2f}",
            "Experimental Binding Affinity (pKd):",
            f"  Min: {min(aff_vals):.3f}, Max: {max(aff_vals):.3f}, Mean: {np.mean(aff_vals):.3f}, Median: {np.median(aff_vals):.3f}, Std: {np.std(aff_vals):.3f}",
            ""
        ])

    report_lines.extend([
        "--- 10. WARNINGS & DROPPED RESIDUES AUDIT LOG ---"
    ])
    if warnings:
        # Include sample of warnings if too long
        for w in warnings[:50]:
            report_lines.append(w)
        if len(warnings) > 50:
            report_lines.append(f"  ... and {len(warnings) - 50} additional residue drop warnings.")
    else:
        report_lines.append("No residue drop warnings encountered.")

    report_lines.extend([
        "",
        "--- 11. SKIPPED / INCOMPATIBLE COMPLEXES AUDIT LOG ---"
    ])
    if failures:
        for f_log in failures:
            report_lines.append(f_log)
    else:
        report_lines.append("Zero failures encountered. All complexes processed successfully.")

    report_lines.extend([
        "",
        "=" * 75,
        "END OF PDBBIND PROCESSING REPORT",
        "=" * 75,
        ""
    ])

    new_content = base_content + "\n" + "\n".join(report_lines)
    with open(info_path, 'w') as f:
        f.write(new_content)


def run_pipeline(
    raw_dir: str = "data/pdbbind/raw",
    processed_dir: str = "data/pdbbind/dataset",
    h5_path: str = "data/pdbbind/dataset.h5",
    index_path: str = "data/pdbbind/index.csv",
    info_path: str = "data/pdbbind/info.txt"
):
    """Executes full PDBbind static tensor processing and HDF5 generation pipeline."""
    print("=" * 80)
    print("STARTING PDBBIND PROTEIN-PROTEIN STATIC PROCESSING PIPELINE (T=1)")
    print("=" * 80)

    # Step 1: Extract static tensors (.pt.gz)
    process_all(raw_dir=raw_dir, processed_dir=processed_dir, index_path=index_path, info_path=info_path)

    # Step 2: Combine to HDF5
    combine_to_h5(processed_dir=processed_dir, h5_path=h5_path, index_path=index_path)

    print("\n" + "=" * 80)
    print("PDBBIND PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"HDF5 Dataset: {h5_path}")
    print(f"Index File:   {index_path}")
    print(f"Summary Log:  {info_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PDBbind Protein-Protein Processing Pipeline")
    parser.add_argument('--process-only', action='store_true', help="Only process PDBs to .pt.gz files")
    parser.add_argument('--combine-only', action='store_true', help="Only combine .pt.gz files to .h5")
    args = parser.parse_args()

    raw_dir = "data/pdbbind/raw"
    processed_dir = "data/pdbbind/dataset"
    h5_path = "data/pdbbind/dataset.h5"
    index_path = "data/pdbbind/index.csv"
    info_path = "data/pdbbind/info.txt"

    if args.process_only:
        process_all(raw_dir=raw_dir, processed_dir=processed_dir, index_path=index_path, info_path=info_path)
    elif args.combine_only:
        combine_to_h5(processed_dir=processed_dir, h5_path=h5_path, index_path=index_path)
    else:
        run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir, h5_path=h5_path, index_path=index_path, info_path=info_path)
