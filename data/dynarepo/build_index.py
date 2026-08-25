import os
import re
import math
import glob
import urllib.request
import csv
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from difflib import SequenceMatcher

RES_MAP_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'MSE': 'M'
}

def get_sequences(pdb_file):
    seqs = {}
    try:
        with open(pdb_file, 'r') as f:
            last_resnum = None
            for line in f:
                if line.startswith('ATOM'):
                    resname = line[17:20].strip()
                    chain = line[21]
                    resnum = line[22:26].strip()
                    
                    # Only take the first atom of each residue to build the sequence
                    if resnum != last_resnum:
                        last_resnum = resnum
                        if resname in RES_MAP_3TO1:
                            if chain not in seqs: seqs[chain] = []
                            seqs[chain].append(RES_MAP_3TO1[resname])
        for c in seqs: seqs[c] = "".join(seqs[c])
        return seqs
    except Exception as e:
        return {}

def seq_sim(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < 0.3: return 0.0
    matches = sum(triple[-1] for triple in sm.get_matching_blocks())
    return matches / min(len(a), len(b))

def load_affinities(index_path):
    aff_map = {}
    affinity_pattern = re.compile(r'(Kd|Ki|IC50)\s*[<>=~]?\s*([\d\.]+)\s*([a-zA-Z]+)', re.IGNORECASE)
    chain_pattern = re.compile(r'\(([a-zA-Z0-9]+)\|([a-zA-Z0-9]+)\)')
    units = {'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9, 'pm': 1e-12, 'fm': 1e-15, 'm': 1}
    
    with open(index_path, 'r') as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 4:
                affinity_match = affinity_pattern.search(parts[3])
                chain_match = chain_pattern.search(line)
                if not affinity_match or not chain_match: continue
                _, val, unit = affinity_match.groups()
                try:
                    val = float(val) * units[unit.lower()]
                    aff_map[parts[0].lower()[:4]] = {
                        'affinity': round(-math.log10(val), 2),
                        'receptor': set(chain_match.group(1)),
                        'ligand': set(chain_match.group(2))
                    }
                except ValueError: continue
    return aff_map

def main():
    aff_map = load_affinities('data/dynarepo/pdbbind/INDEX_general_PP.2020R1.lst')
    rcsb_dir = 'data/dynarepo/rcsb'
    os.makedirs(rcsb_dir, exist_ok=True)
    
    out_csv = 'data/dynarepo/index.csv'
    
    pdb_files = glob.glob('data/dynarepo/raw/*.pdb')
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dynarepo_id', 'pdb_id', 'affinity', 'receptor', 'ligand'])
        
        seen_dynarepo_ids = set()
        for sim_pdb in sorted(pdb_files):
            if 'rcsb' in sim_pdb: continue
            
            basename = os.path.basename(sim_pdb)
            dynarepo_id = basename.split('.')[0]
            if dynarepo_id in seen_dynarepo_ids: continue
            
            pdb_id = basename.split('_')[1].split('.')[0].lower()
            if pdb_id not in aff_map: continue
            
            aff_data = aff_map[pdb_id]
            ref_receptor = aff_data['receptor']
            ref_ligand = aff_data['ligand']
            
            rcsb_pdb = os.path.join(rcsb_dir, f"{pdb_id}.pdb")
            if not os.path.exists(rcsb_pdb):
                try:
                    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", rcsb_pdb)
                except Exception as e:
                    print(f"Failed to download {pdb_id}: {e}")
                    continue
            
            ref_seqs = get_sequences(rcsb_pdb)
            sim_seqs = get_sequences(sim_pdb)
            
            expanded_rec = set()
            expanded_lig = set()
            for c_ref, seq_ref in ref_seqs.items():
                if any(seq_sim(seq_ref, ref_seqs[r]) > 0.95 for r in ref_receptor if r in ref_seqs):
                    expanded_rec.add(c_ref)
                if any(seq_sim(seq_ref, ref_seqs[l]) > 0.95 for l in ref_ligand if l in ref_seqs):
                    expanded_lig.add(c_ref)
            
            sim_to_ref = {}
            for s_chain, s_seq in sim_seqs.items():
                best_ref = None
                best_score = 0
                for r_chain, r_seq in ref_seqs.items():
                    score = seq_sim(s_seq, r_seq)
                    if score > best_score:
                        best_score = score
                        best_ref = r_chain
                if best_score > 0.3:
                    sim_to_ref[s_chain] = best_ref
            
            sim_rec_cands = [s for s, r in sim_to_ref.items() if r in expanded_rec]
            sim_lig_cands = [s for s, r in sim_to_ref.items() if r in expanded_lig]
            
            def get_best_interface(sim_pdb, sim_rec_cands, sim_lig_cands, n_rec, n_lig):
                coords = {}
                with open(sim_pdb, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM'):
                            chain = line[21]
                            if chain not in coords: coords[chain] = []
                            coords[chain].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                
                for c in coords: coords[c] = np.array(coords[c])
                
                best_dist = float('inf')
                best_rec, best_lig = None, None
                
                if len(sim_rec_cands) < n_rec or len(sim_lig_cands) < n_lig:
                    return None, None
                    
                for r_combo in itertools.combinations(sim_rec_cands, n_rec):
                    for l_combo in itertools.combinations(sim_lig_cands, n_lig):
                        if set(r_combo) & set(l_combo): continue
                        r_coords = [coords[c] for c in r_combo if c in coords]
                        l_coords = [coords[c] for c in l_combo if c in coords]
                        if not r_coords or not l_coords: continue
                        
                        min_d = np.min(cdist(np.vstack(r_coords), np.vstack(l_coords)))
                        if min_d < best_dist:
                            best_dist = min_d
                            best_rec = r_combo
                            best_lig = l_combo
                            
                if best_dist > 5.0: return None, None
                return best_rec, best_lig

            mapped_rec_tuple, mapped_lig_tuple = get_best_interface(
                sim_pdb, sim_rec_cands, sim_lig_cands, len(ref_receptor), len(ref_ligand)
            )
            
            if mapped_rec_tuple is None or mapped_lig_tuple is None:
                print(f"[\u274c] {basename}: Failed mapping. PDBbind={ref_receptor}|{ref_ligand}, SimChains={list(sim_seqs.keys())}")
                continue
                
            mapped_receptor = set(mapped_rec_tuple)
            mapped_ligand = set(mapped_lig_tuple)
                
            writer.writerow([
                dynarepo_id,
                pdb_id, 
                aff_data['affinity'], 
                "".join(sorted(list(mapped_receptor))), 
                "".join(sorted(list(mapped_ligand)))
            ])
            seen_dynarepo_ids.add(dynarepo_id)

if __name__ == '__main__':
    main()
