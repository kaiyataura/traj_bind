import os
import re
import math
import gzip
import warnings
import requests
import concurrent.futures
from glob import glob
import numpy as np
import csv
import torch
from tqdm import tqdm
import MDAnalysis as mda
import h5py

BASE_URL = "https://dynarepo.inria.fr/api/rest/v1"

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

def load_index(index_path: str) -> list:
    index = []
    with open(index_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index.append({
                'dynarepo_id': row['dynarepo_id'],
                'pdb_id': row['pdb_id'],
                'affinity': float(row['affinity']),
                'receptor': set(row['receptor']),
                'ligand': set(row['ligand'])
            })
    return index


def download_one(output_dir: str, accession: str, frames: int, pdb_id: str) -> str:
    pdb_out = os.path.join(output_dir, f"{accession}_{pdb_id}.pdb")
    xtc_out = os.path.join(output_dir, f"{accession}_{pdb_id}.xtc")
    
    if os.path.exists(xtc_out) and os.path.exists(pdb_out):
        return f"  [\u2714] [Skip] {accession} ({pdb_id}) already exists."

    try:
        sel = "protein%20and%20not%20hydrogen"
        with open(pdb_out, 'wb') as f:
            f.write(requests.get(f'{BASE_URL}/projects/{accession}/files/structure?selection={sel}', stream=True).content)
        with open(xtc_out, 'wb') as f:
            resp = requests.get(f'{BASE_URL}/projects/{accession}/files/trajectory?selection={sel}&frames=1:{frames}:1&format=xtc', stream=True)
            if resp.status_code == 400:
                affinity_match = re.search(r'beyond the limit \((\d+)\)', resp.text)
                if not affinity_match: return f"  [\u274c] [Error] Failed to download {accession} ({pdb_id}): {resp.text}"
                frames = int(affinity_match.group(1))
                resp = requests.get(f'{BASE_URL}/projects/{accession}/files/trajectory?selection={sel}&frames=1:{frames}:1&format=xtc', stream=True)
            f.write(resp.content)
        return f"  [\u2714] Downloaded {accession} ({pdb_id})"
    except Exception as e:
        return f"  [\u274c] [Error] Failed on {accession} ({pdb_id}): {e}"


def download(output_dir, index_path):
    os.makedirs(output_dir, exist_ok=True)
    index_map = load_index(index_path)
    
    if not index_map:
        print("Fatal: No affinities loaded. Check your index file path.")
        return

    print("Fetching projects from DynaRepo (handling pagination)...")
    projects = []
    page = 1
    limit = 100
    
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/projects?limit={limit}&page={page}")
            resp.raise_for_status()
            
            data = resp.json()['projects']
            if not data: break
                
            projects.extend(data)
            page += 1
            print(f"  Fetched {len(projects)} projects so far...")
            
        except Exception as e:
            print(f"Failed to fetch projects at page={page}: {e}")
            return
    
    print(f"\nTotal projects to evaluate: {len(projects)}")

    tasks = []
    for project in projects:
        accession = project['accession']
        metadata = project['metadata']
        pdb_ids = metadata['PDBIDS']
        if len(pdb_ids) != 1: continue
        pdb_id = str(pdb_ids[0]).lower()
        if pdb_id not in index_map: continue
        protseq = metadata['PROTSEQ']
        if len(protseq) < 2: continue
        if len(metadata['NUCLSEQ']) > 0: continue
        frames = metadata['mdFrames']
        
        mds = project['mds']
        for i in range(len(mds)):
            tasks.append((output_dir, f'{accession}.{i + 1}', frames, pdb_id))

    print(f"Queueing {len(tasks)} replica tasks for parallel execution...\n")

    max_workers = min(os.cpu_count() or 4, 16) 
    print(f"Starting ProcessPoolExecutor with {max_workers} CPU cores...\n")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, *task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Processing Replicas", unit="traj") as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_info = futures[future]
                try:
                    result_msg = future.result()
                    tqdm.write(result_msg)
                except Exception as exc:
                    print(f"  [\u274c] Fatal parallel error on {task_info}: {exc}")
                finally:
                    pbar.update(1)


def process_one(pdb_path: str, xtc_paths: list[str], metadata: dict, processed_dir: str) -> tuple:
    warnings = []
    try:
        import pyrosetta
        u = mda.Universe(pdb_path, *xtc_paths)
        
        original_resnames = u.residues.resnames.copy()
        temp_resnames = original_resnames.copy()
        
        # PyRosetta doesn't like non-standard protonation states in PDBs.
        # It's better to give it standard names and let it repack them.
        rosetta_aliases = {
            'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS', 'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
            'ASH': 'ASP', 'GLH': 'GLU', 'LYN': 'LYS', 'TYM': 'TYR', 'CYX': 'CYS', 'CYM': 'CYS',
            'AR0': 'ARG', 'MSE': 'MET'
        }
        for i, name in enumerate(temp_resnames):
            name_upper = name.upper()
            if name_upper in rosetta_aliases:
                temp_resnames[i] = rosetta_aliases[name_upper]
        u.residues.resnames = temp_resnames
        
        tmp_pdb = pdb_path.replace('.pdb', f'_{os.getpid()}_pr.pdb')
        u.select_atoms("protein").write(tmp_pdb)
        
        # Restore original names so md_res gets the correct protonation states
        u.residues.resnames = original_resnames
        
        pose = pyrosetta.pose_from_pdb(tmp_pdb)
        if os.path.exists(tmp_pdb): os.remove(tmp_pdb)
        
        scorefxn = pyrosetta.get_fa_scorefxn()
        scorefxn(pose)
        
        md_pos = u.atoms.positions # type: ignore
        
        md_res = []
        for res in u.select_atoms("protein").residues:
            name = RES_ALIASES.get(res.resname.upper(), res.resname.upper())
            if name not in RES_MAP: continue
            
            atoms = {a.name: a.index for a in res.atoms}
            if not {'N', 'CA', 'C'}.issubset(atoms.keys()): 
                warnings.append(f"  [\u26A0] [Warning] {os.path.basename(pdb_path)}: Dropping {name} {res.resnum} missing backbone.")
                continue
            
            sc = [a.index for a in res.atoms if a.name not in {'C', 'CA', 'N', 'O', 'HN', 'HA', 'H'}]
            md_res.append({
                'id': RES_MAP[name], 'props': RES_PROPERTIES[name],
                'ca': atoms['CA'], 'c': atoms['C'], 'n': atoms['N'], 'sc': sc if sc else [atoms['CA']],
                'name': name, 'resnum': res.resnum
            })
            
        if not md_res: return None, warnings + [f"  [\u26A0] [Skip] {os.path.basename(pdb_path)}: No MD residues."]

        pr_ca = np.array([pose.residue(i).xyz("CA") if pose.residue(i).has("CA") else [np.inf]*3 for i in range(1, pose.total_residue() + 1)])
        if len(pr_ca) == 0: return None, warnings + [f"  [\u26A0] [Skip] {os.path.basename(pdb_path)}: No PyRosetta CA atoms."]
        
        md_ca = np.array([md_pos[r['ca']] for r in md_res])
        dists = np.linalg.norm(md_ca[:, None, :] - pr_ca[None, :, :], axis=2)
        
        warnings = []
        dropped_residues = 0
        aligned_res = []
        for i, r in enumerate(md_res):
            min_dist = dists[i].min()
            if min_dist > 0.1:
                dropped_residues += 1
                continue
            
            pr_res = dists[i].argmin() + 1
            pr_chain = pose.pdb_info().chain(pr_res)
            if pr_chain in metadata['receptor']: partner = 0
            elif pr_chain in metadata['ligand']: partner = 1
            else: continue
            
            r['chain'] = partner
            r['pr_res'] = pr_res
            aligned_res.append(r)
            
        if dropped_residues > 0:
            warnings.append(f"  [\u26A0] [Warning] {os.path.basename(pdb_path)}: Dropped {dropped_residues} residues that were missing in the MD trajectory.")

        if not aligned_res: return None, warnings + [f"  [\u26A0] [Skip] {os.path.basename(pdb_path)}: Alignment empty."]

        sc_coords = np.array([md_pos[r['sc']].mean(axis=0) for r in aligned_res])
        chains = np.array([r['chain'] for r in aligned_res])
        
        if 0 not in chains or 1 not in chains:
            return None, warnings + [f"  [\u26A0] [Skip] {os.path.basename(pdb_path)}: Missing a partner."]
            
        sc_dists = np.linalg.norm(sc_coords[:, None, :] - sc_coords[None, :, :], axis=2)
        min_dists = np.where(chains == 0, sc_dists[:, chains == 1].min(axis=1), sc_dists[:, chains == 0].min(axis=1))
        
        keep_mask = min_dists <= 25.0
        core_mask = min_dists[keep_mask] <= 15.0
        valid_res = [aligned_res[i] for i in range(len(aligned_res)) if keep_mask[i]]
        N = len(valid_res)
        
        if N == 0: return None, warnings + [f"  [\u26A0] [Skip] {os.path.basename(pdb_path)}: Interface empty."]

        valid_pr_res = {r['pr_res'] for r in valid_res}
        md_to_pr = []
        for i in valid_pr_res:
            for j in range(1, pose.residue(i).natoms() + 1):
                if pose.residue(i).atom_is_hydrogen(j): continue
                xyz = pose.residue(i).xyz(j)
                atom_dists = np.linalg.norm(md_pos - np.array(xyz), axis=1)
                if atom_dists.min() < 0.1: md_to_pr.append((np.argmin(atom_dists), i, j))

        e_types = [pyrosetta.rosetta.core.scoring.fa_atr, pyrosetta.rosetta.core.scoring.fa_rep, pyrosetta.rosetta.core.scoring.fa_elec, pyrosetta.rosetta.core.scoring.fa_sol, 
                   pyrosetta.rosetta.core.scoring.hbond_sr_bb, pyrosetta.rosetta.core.scoring.hbond_lr_bb, pyrosetta.rosetta.core.scoring.hbond_bb_sc, pyrosetta.rosetta.core.scoring.hbond_sc]
        
        ca_idx = [r['ca'] for r in valid_res]
        c_idx = [r['c'] for r in valid_res]
        n_idx = [r['n'] for r in valid_res]
        
        all_coords, all_energies = [], []
        for _ in u.trajectory:
            pos = u.atoms.positions # type: ignore
            for md_idx, pr_res, pr_atom in md_to_pr:
                pose.residue(pr_res).set_xyz(pr_atom, pyrosetta.rosetta.numeric.xyzVector_double_t(*pos[md_idx]))
            scorefxn(pose)
            
            frame_c = np.zeros((N, 4, 3), dtype=np.float32)
            frame_c[:, 0], frame_c[:, 1], frame_c[:, 2] = pos[ca_idx], pos[c_idx], pos[n_idx]
            for i, r in enumerate(valid_res): frame_c[i, 3] = pos[r['sc']].mean(axis=0)
            all_coords.append(frame_c)
            
            sc_pos = frame_c[:, 3]
            dists = np.linalg.norm(sc_pos[:, None, :] - sc_pos[None, :, :], axis=2)
            
            frame_e = np.zeros((N, N, 4), dtype=np.float32)
            for i in range(N):
                if not core_mask[i]: continue
                for j in range(i+1, N):
                    if not core_mask[j]: continue
                    if dists[i, j] > 20.0: continue
                    pr_i, pr_j = pose.residue(valid_res[i]['pr_res']), pose.residue(valid_res[j]['pr_res'])
                    emap = pyrosetta.rosetta.core.scoring.EMapVector()
                    scorefxn.eval_ci_2b(pr_i, pr_j, pose, emap)
                    scorefxn.eval_cd_2b(pr_i, pr_j, pose, emap)
                    frame_e[i, j] = frame_e[j, i] = [emap[e_types[0]] + emap[e_types[1]], emap[e_types[2]], emap[e_types[3]], sum(emap[t] for t in e_types[4:])]
            all_energies.append(frame_e)
            
        dynarepo_id = metadata['dynarepo_id']
        
        with gzip.open(os.path.join(processed_dir, f"{dynarepo_id}.pt.gz"), 'wb') as f:
            torch.save({
                'T': len(u.trajectory),
                'N': N,
                'dynarepo_id': dynarepo_id,
                'affinity': metadata['affinity'],
                'coords': torch.from_numpy(np.stack(all_coords).astype(np.float16)),
                'energies': torch.from_numpy(np.stack(all_energies).astype(np.float16)),
                'residues': torch.from_numpy(np.array([r['id'] for r in valid_res], dtype=np.int8)),
                'props': torch.from_numpy(np.array([r['props'] for r in valid_res], dtype=np.float32)),
                'chain_ids': torch.from_numpy(np.array([r['chain'] for r in valid_res], dtype=np.int8)),
                'core_mask': torch.from_numpy(core_mask)
            }, f)
        
        return True, warnings + [f"  [\u2714] Processed {os.path.basename(pdb_path)} (T={len(u.trajectory)}, N={N})"]
        
    except Exception as e:
        return False, warnings + [f"  [\u274c] [Error] Failed on {os.path.basename(pdb_path)}: {e}"]


def init_worker():
    import sys, os
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false -load_PDB_components false")


def process(input_dir: str, output_path: str, index_path: str):
    index_list = load_index(index_path)
    os.makedirs(output_path, exist_ok=True)
    
    tasks = []
    for info in index_list:
        prefix = info['dynarepo_id']
        pdb_cands = glob(os.path.join(input_dir, f"{prefix}.*.pdb"))
        if not pdb_cands: continue
        base_pdb = sorted(pdb_cands)[0]
        
        xtcs = glob(os.path.join(input_dir, f"{prefix}.*.xtc"))
        if xtcs:
            tasks.append((base_pdb, xtcs, info, output_path))

    print(f"Queueing {len(tasks)} complexes for tensor extraction...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 16), initializer=init_worker) as executor:
        futures = {executor.submit(process_one, *t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Extracting Trajectories"):
            try:
                success, warnings = future.result()
                for w in warnings:
                    if '[\u274c]' in w or '[\u26A0]' in w: tqdm.write(w)
            except Exception as e:
                print(f"Extraction failed: {e}")
            
    print(f"\nSuccessfully extracted trajectories to {output_path}.")
    
def combine_to_h5(processed_dir, h5_path):
    pt_files = sorted(glob(os.path.join(processed_dir, "*.pt.gz")))
    if not pt_files:
        print("No .pt.gz files found to combine.")
        return

    print(f"\nCombining {len(pt_files)} files into {h5_path}...")
    with h5py.File(h5_path, 'w') as h5f:
        for pt_file in tqdm(pt_files, desc="Converting to HDF5"):
            with gzip.open(pt_file, 'rb') as f:
                data = torch.load(f, weights_only=False)
            
            dynarepo_id = data['dynarepo_id']
            N = data['N']
            
            grp = h5f.create_group(dynarepo_id)
            grp.attrs['T'] = data['T']
            grp.attrs['N'] = N
            grp.attrs['dynarepo_id'] = dynarepo_id
            grp.attrs['affinity'] = data['affinity']
            
            grp.create_dataset('coords', data=data['coords'].numpy(), chunks=(1, N, 4, 3), compression='gzip', compression_opts=4)
            grp.create_dataset('energies', data=data['energies'].numpy(), chunks=(1, N, N, 4), compression='gzip', compression_opts=4)
            grp.create_dataset('residues', data=data['residues'].numpy())
            grp.create_dataset('props', data=data['props'].numpy())
            grp.create_dataset('chain_ids', data=data['chain_ids'].numpy())
            grp.create_dataset('core_mask', data=data['core_mask'].numpy())
            
    print(f"Successfully created single HDF5 dataset at {h5_path}")

if __name__ == '__main__':
    # process("data/dynarepo/raw", "data/dynarepo/dataset", "data/dynarepo/index.csv")
    combine_to_h5("data/dynarepo/dataset", "data/dynarepo/dataset.h5")
