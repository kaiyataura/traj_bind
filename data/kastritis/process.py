import os
import io
import csv
import gzip
import tarfile
import argparse
import requests
import numpy as np
import pandas as pd
import torch
import h5py
import MDAnalysis as mda
from glob import glob
from tqdm import tqdm
import concurrent.futures

PRODIGY_CSV_URL = "https://wenmr.science.uu.nl/prodigy/static/PRODIGY_dataset.csv"
PRODIGY_TGZ_URL = "https://wenmr.science.uu.nl/prodigy/static/PRODIGYdataset.tgz"

RES_MAP = {
    'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 'GLU': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10,
    'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20,
    'HSD': 21, 'HSE': 22, 'HSP': 23, 'ASH': 24, 'GLH': 25, 'LYN': 26, 'TYM': 27, 'CYX': 28, 'CYM': 29 
}
RES_ALIASES = { 'HID': 'HSD', 'HIE': 'HSE', 'HIP': 'HSP', 'AR0': 'ARG', 'MSE': 'MET' }

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


def download(output_raw_dir: str = "data/kastritis/raw", index_path: str = "data/kastritis/index.csv") -> pd.DataFrame:
    os.makedirs(output_raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)

    resp_csv = requests.get(PRODIGY_CSV_URL, timeout=30)
    resp_csv.raise_for_status()
    df = pd.read_csv(io.StringIO(resp_csv.text))

    resp_tgz = requests.get(PRODIGY_TGZ_URL, timeout=60)
    resp_tgz.raise_for_status()

    try: tar = tarfile.open(fileobj=io.BytesIO(resp_tgz.content), mode='r:*')
    except Exception as e: raise RuntimeError(f"Failed to read tar archive: {e}")

    pdb_members = [
        m for m in tar.getmembers()
        if m.name.endswith('.pdb') and not os.path.basename(m.name).startswith('._')
    ]
    
    for member in pdb_members:
        filename = os.path.basename(member.name)
        target_path = os.path.join(output_raw_dir, filename)
        f_in = tar.extractfile(member)
        if f_in is None: continue
        with open(target_path, 'wb') as f_out: f_out.write(f_in.read())

    records = []
    for _, row in df.iterrows():
        pdb_file = row['PDB']
        pdb_id = pdb_file.replace('.pdb', '').lower()
        dg = float(row['DG'])
        pkd = -dg / 1.3633 # pKd = -DG / (RT * ln(10)) = -DG / 1.3633 at 298.15 K
        chains_str = str(row['Interacting_chains'])
        rec_orig, lig_orig = chains_str.split(':') if ':' in chains_str else ('A', 'B')

        records.append({
            'complex_id': pdb_id,
            'pdb_id': pdb_id,
            'pdb_file': pdb_file,
            'affinity': pkd,
            'delta_g': dg,
            'receptor': 'A',      # PRODIGY clean format standardizes receptor to Chain A
            'ligand': 'B',        # PRODIGY clean format standardizes ligand to Chain B
            'orig_chains': chains_str,
            'functional_class': row.get('Functional_class', 'N/A'),
            'exp_method': row.get('Experimental_method', 'N/A')
        })

    index_df = pd.DataFrame(records)
    index_df.to_csv(index_path, index=False)
    return index_df


def load_index(index_path: str) -> list:
    index = []
    with open(index_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index.append({
                'complex_id': row['complex_id'],
                'pdb_id': row['pdb_id'],
                'pdb_file': row.get('pdb_file', f"{row['pdb_id'].upper()}.pdb"),
                'affinity': float(row['affinity']),
                'delta_g': float(row['delta_g']),
                'receptor': set(row['receptor']),
                'ligand': set(row['ligand']),
                'orig_chains': row.get('orig_chains', 'A:B'),
                'functional_class': row.get('functional_class', 'N/A'),
                'exp_method': row.get('exp_method', 'N/A')
            })
    return index


def process_one(pdb_path: str, metadata: dict, processed_dir: str) -> tuple:
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
            
            # Segid / Chain ID in MDAnalysis
            segid = res.segid if res.segid else (res.atoms.segids[0] if len(res.atoms.segids) > 0 else 'A')
            if segid in metadata['receptor']:
                partner = 0
            elif segid in metadata['ligand']:
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
                'segid': segid
            })

        if not md_res:
            return False, {}, warnings + [f"  [❌] [Skip] {complex_id}: No valid MD residues found."]

        chains = np.array([r['chain'] for r in md_res])
        if 0 not in chains or 1 not in chains:
            return False, {}, warnings + [f"  [❌] [Skip] {complex_id}: Missing a binding partner (chains={set(chains)})."]

        # Compute pairwise sidechain centroid cross-partner distances
        sc_coords = np.array([pos[r['sc']].mean(axis=0) for r in md_res])
        sc_dists = np.linalg.norm(sc_coords[:, None, :] - sc_coords[None, :, :], axis=2)
        min_dists = np.where(
            chains == 0,
            sc_dists[:, chains == 1].min(axis=1),
            sc_dists[:, chains == 0].min(axis=1)
        )

        # Distance thresholding: keep within 25 Å, core within 15 Å
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
            'delta_g': float(metadata['delta_g']),
            'functional_class': metadata.get('functional_class', 'N/A'),
            'warnings': warnings
        }
        return True, stat_info, warnings + [f"  [✔] Processed {complex_id} (T=1, N={N}, core={int(core_mask.sum())})"]

    except Exception as e:
        return False, {}, warnings + [f"  [❌] [Error] Failed on {complex_id}: {e}"]


def process_all(
    raw_dir: str = "data/kastritis/raw",
    processed_dir: str = "data/kastritis/dataset",
    index_path: str = "data/kastritis/index.csv",
    info_path: str = "data/kastritis/info.txt"
) -> list:
    """
    Processes all benchmark complexes in parallel, extracts intermediate .pt.gz files,
    and writes comprehensive info.txt log.
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

    print(f"\nProcessing {len(tasks)} benchmark complexes for static tensor extraction (T=1)...")

    max_workers = min(os.cpu_count() or 4, 16)
    successful_stats = []
    failed_logs = []
    all_warnings = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_one, *t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Extracting Benchmark"):
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

    # Write info.txt documentation
    write_info_txt(info_path, len(index_records), successful_stats, failed_logs, all_warnings)
    print(f"\nSuccessfully extracted {len(successful_stats)} / {len(tasks)} complexes to {processed_dir}")
    print(f"Processing documentation written to {info_path}")
    return successful_stats


def combine_to_h5(
    processed_dir: str = "data/kastritis/dataset",
    h5_path: str = "data/kastritis/dataset.h5"
):
    """
    Combines intermediate .pt.gz files into final HDF5 dataset conforming to TrajectoryDataset schema.
    """
    pt_files = sorted(glob(os.path.join(processed_dir, "*.pt.gz")))
    if not pt_files:
        raise RuntimeError(f"No .pt.gz files found in {processed_dir} to combine into HDF5.")

    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    print(f"\nCombining {len(pt_files)} intermediate tensor files into {h5_path}...")

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


def write_info_txt(
    info_path: str,
    total_requested: int,
    stats: list,
    failures: list,
    warnings: list
):
    """Writes detailed info.txt log documenting processing summary and statistics."""
    with open(info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("KASTRITIS / PRODIGY PROTEIN-PROTEIN AFFINITY BENCHMARK PROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET SOURCE & SPECIFICATIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Source URL (CSV): {PRODIGY_CSV_URL}\n")
        f.write(f"Source URL (PDBs): {PRODIGY_TGZ_URL}\n")
        f.write("Benchmark Reference: Vangone & Bonvin, eLife 2015, 4:e07454; Kastritis et al., Protein Sci 2011\n")
        f.write("Temporal Dimension: T = 1 (Static crystal structures)\n")
        f.write("Affinity Formulation: pKd = -DeltaG / (RT * ln(10)) = -DeltaG / 1.3633 (kcal/mol at 298.15 K)\n")
        f.write("Interface Definition: Interface residues with cross-partner min sidechain dist <= 25.0 A\n")
        f.write("Core Interface Definition: Interface residues with cross-partner min sidechain dist <= 15.0 A\n")
        f.write("Coordinates: [1, N, 4, 3] float16 (CA=0, C=1, N=2, Sidechain Centroid=3)\n\n")

        f.write("2. SUMMARY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total complexes in benchmark index: {total_requested}\n")
        f.write(f"Successfully processed complexes:   {len(stats)}\n")
        f.write(f"Failed / skipped complexes:         {len(failures)}\n")
        f.write(f"Total warnings logged:              {len(warnings)}\n\n")

        if stats:
            n_vals = [s['N'] for s in stats]
            core_vals = [s['core_N'] for s in stats]
            aff_vals = [s['affinity'] for s in stats]
            dg_vals = [s['delta_g'] for s in stats]

            f.write("Residue Count (N) Statistics:\n")
            f.write(f"  Min: {min(n_vals)}, Max: {max(n_vals)}, Mean: {np.mean(n_vals):.2f}, Median: {np.median(n_vals):.1f}\n")
            f.write("Core Interface Residues (dist <= 15A) Statistics:\n")
            f.write(f"  Min: {min(core_vals)}, Max: {max(core_vals)}, Mean: {np.mean(core_vals):.2f}, Median: {np.median(core_vals):.1f}\n")
            f.write("Experimental Affinity (pKd) Statistics:\n")
            f.write(f"  Min: {min(aff_vals):.3f}, Max: {max(aff_vals):.3f}, Mean: {np.mean(aff_vals):.3f}, Std: {np.std(aff_vals):.3f}\n")
            f.write("Binding Free Energy (DeltaG kcal/mol) Statistics:\n")
            f.write(f"  Min: {min(dg_vals):.2f}, Max: {max(dg_vals):.2f}, Mean: {np.mean(dg_vals):.2f}, Std: {np.std(dg_vals):.2f}\n\n")

        f.write("3. PROCESSED COMPLEXES DETAILS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Complex':<10} {'PDB':<8} {'Class':<8} {'N':<6} {'Core_N':<8} {'Rec_N':<8} {'Lig_N':<8} {'DeltaG':<10} {'pKd':<8}\n")
        f.write("-" * 80 + "\n")
        for s in stats:
            f.write(
                f"{s['complex_id']:<10} {s['pdb_id']:<8} {s['functional_class']:<8} "
                f"{s['N']:<6} {s['core_N']:<8} {s['rec_N']:<8} {s['lig_N']:<8} "
                f"{s['delta_g']:<10.2f} {s['affinity']:<8.3f}\n"
            )
        f.write("\n")

        f.write("4. WARNINGS & DROPPED RESIDUES LOG\n")
        f.write("-" * 40 + "\n")
        if warnings:
            for w in warnings:
                f.write(f"{w}\n")
        else:
            f.write("No residue drop warnings encountered.\n")
        f.write("\n")

        f.write("5. FAILURES LOG\n")
        f.write("-" * 40 + "\n")
        if failures:
            for fail in failures:
                f.write(f"{fail}\n")
        else:
            f.write("Zero processing failures encountered. All benchmark complexes succeeded.\n")


def run_pipeline(
    raw_dir: str = "data/kastritis/raw",
    processed_dir: str = "data/kastritis/dataset",
    h5_path: str = "data/kastritis/dataset.h5",
    index_path: str = "data/kastritis/index.csv",
    info_path: str = "data/kastritis/info.txt"
):
    """Executes full benchmark acquisition and processing pipeline."""
    print("=" * 80)
    print("STARTING KASTRITIS AFFINITY BENCHMARK PROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Download & Index
    download(output_raw_dir=raw_dir, index_path=index_path)

    # Step 2: Extract static tensors (.pt.gz)
    process_all(raw_dir=raw_dir, processed_dir=processed_dir, index_path=index_path, info_path=info_path)

    # Step 3: Combine to HDF5
    combine_to_h5(processed_dir=processed_dir, h5_path=h5_path)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"HDF5 Dataset: {h5_path}")
    print(f"Index File:   {index_path}")
    print(f"Summary Log:  {info_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kastritis Benchmark Pipeline")
    parser.add_argument('--download-only', action='store_true', help="Only download benchmark raw data")
    parser.add_argument('--process-only', action='store_true', help="Only process extracted PDBs")
    parser.add_argument('--combine-only', action='store_true', help="Only combine .pt.gz to .h5")
    args = parser.parse_args()

    raw_dir = "data/kastritis/raw"
    processed_dir = "data/kastritis/dataset"
    h5_path = "data/kastritis/dataset.h5"
    index_path = "data/kastritis/index.csv"
    info_path = "data/kastritis/info.txt"

    if args.download_only:
        download(output_raw_dir=raw_dir, index_path=index_path)
    elif args.process_only:
        process_all(raw_dir=raw_dir, processed_dir=processed_dir, index_path=index_path, info_path=info_path)
    elif args.combine_only:
        combine_to_h5(processed_dir=processed_dir, h5_path=h5_path)
    else:
        run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir, h5_path=h5_path, index_path=index_path, info_path=info_path)
