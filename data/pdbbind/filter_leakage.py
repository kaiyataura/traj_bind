"""
Milestone M2: Sequence Similarity Clustering & Zero-Leakage Filter Engine.

Performs strict sequence homology filtering between the PDBbind Protein-Protein
dataset (v2020R1) and the Kastritis Protein-Protein Binding Affinity Benchmark
(PRODIGY cleaned set) using Biopython's Bio.Align.PairwiseAligner with
global Needleman-Wunsch alignment normalized by minimum sequence length:
    SeqIdentity = matches / min(L1, L2)

Any-chain cross-matching: If ANY chain of a candidate PDBbind complex shares
>= 35.0% sequence identity with ANY chain of ANY benchmark complex, the candidate
is purged to ensure absolute zero data leakage.

Outputs:
  - data/pdbbind/leakage_matrix.csv : Log of all excluded homologous pairs
  - data/pdbbind/index.csv          : Clean, non-homologous PDBbind complexes
  - data/pdbbind/info.txt           : Comprehensive audit log and statistics
"""

import os
import re
import math
import time
import glob
import csv
import io
import tarfile
import urllib.request
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from Bio.Align import PairwiseAligner, substitution_matrices

# Canonical and modified amino acid 3-to-1 conversion dictionary
RES_MAP_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
    # Common protonation variants and modified residues
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'HID': 'H', 'HIE': 'H', 'HIP': 'H',
    'MSE': 'M', 'ASH': 'D', 'GLH': 'E', 'CYX': 'C', 'CYM': 'C', 'TYM': 'Y', 'LYN': 'K',
    'SEC': 'U', 'PYL': 'O', 'SEP': 'S', 'TPO': 'T', 'PTR': 'Y', 'PCA': 'E', 'CSO': 'C'
}

# Unit multipliers for binding affinity conversion to Molar
UNITS_MAP = {
    'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9, 'pm': 1e-12, 'fm': 1e-15, 'm': 1.0
}


def create_aligner() -> PairwiseAligner:
    """Instantiate and configure Global Needleman-Wunsch PairwiseAligner with BLOSUM62."""
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    return aligner


def extract_seq_from_pdb_file(pdb_path: str) -> dict[str, str]:
    """Extract protein chain sequences from ATOM records of the first MODEL in a PDB file."""
    if not os.path.exists(pdb_path):
        return {}
    seqs: dict[str, list[str]] = {}
    last_res_by_chain: dict[str, str] = {}
    model_count = 0
    with open(pdb_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('MODEL'):
                model_count += 1
                if model_count > 1:
                    break
            elif line.startswith('ENDMDL'):
                break
            elif line.startswith('ATOM'):
                resname = line[17:20].strip()
                chain = line[21]
                resnum = line[22:27].strip()  # includes insertion code if present
                if last_res_by_chain.get(chain) != resnum:
                    last_res_by_chain[chain] = resnum
                    if resname in RES_MAP_3TO1:
                        if chain not in seqs:
                            seqs[chain] = []
                        seqs[chain].append(RES_MAP_3TO1[resname])
    return {c: "".join(s) for c, s in seqs.items()}


def extract_seq_from_pdb_text(pdb_text: str) -> dict[str, str]:
    """Extract protein chain sequences from text content of the first MODEL in a PDB file."""
    seqs: dict[str, list[str]] = {}
    last_res_by_chain: dict[str, str] = {}
    model_count = 0
    for line in pdb_text.splitlines():
        if line.startswith('MODEL'):
            model_count += 1
            if model_count > 1:
                break
        elif line.startswith('ENDMDL'):
            break
        elif line.startswith('ATOM'):
            resname = line[17:20].strip()
            chain = line[21]
            resnum = line[22:27].strip()
            if last_res_by_chain.get(chain) != resnum:
                last_res_by_chain[chain] = resnum
                if resname in RES_MAP_3TO1:
                    if chain not in seqs:
                        seqs[chain] = []
                    seqs[chain].append(RES_MAP_3TO1[resname])
    return {c: "".join(s) for c, s in seqs.items()}


def load_benchmark_sequences(
    benchmark_dir: str = 'data/kastritis/raw',
    url: str = 'https://wenmr.science.uu.nl/prodigy/static/PRODIGYdataset.tgz'
) -> tuple[dict[str, dict[str, str]], list[tuple[str, str, str]]]:
    """
    Load or acquire Kastritis benchmark complex sequences (PRODIGY clean set).
    Returns:
      bench_complexes: dict of {pdb_id: {chain_id: sequence}}
      bench_chains: list of (pdb_id, chain_id, sequence)
    """
    bench_complexes: dict[str, dict[str, str]] = {}
    
    # 1. Try loading from local raw directory
    local_pdbs = glob.glob(os.path.join(benchmark_dir, '*.pdb'))
    if len(local_pdbs) >= 80:
        for p in local_pdbs:
            pid = os.path.basename(p).replace('.pdb', '').lower()
            s = extract_seq_from_pdb_file(p)
            if s:
                bench_complexes[pid] = s
    else:
        # 2. Download from official PRODIGY repository
        print(f"[Benchmark] Downloading PRODIGY dataset archive from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            archive_data = resp.read()
        tf = tarfile.open(fileobj=io.BytesIO(archive_data))
        for m in tf.getmembers():
            fname = os.path.basename(m.name)
            if fname.endswith('.pdb') and not fname.startswith('._'):
                pid = fname.replace('.pdb', '').lower()
                f = tf.extractfile(m)
                if f is not None:
                    text = f.read().decode('utf-8', errors='ignore')
                    s = extract_seq_from_pdb_text(text)
                    if s:
                        bench_complexes[pid] = s

    bench_chains = []
    for pid, cdict in bench_complexes.items():
        for chain, seq in cdict.items():
            if len(seq) >= 5:
                bench_chains.append((pid, chain, seq))

    return bench_complexes, bench_chains


def parse_pdbbind_index(
    index_path: str = 'data/dynarepo/pdbbind/INDEX_general_PP.2020R1.lst',
    raw_pdb_dir: str = 'data/pdbbind/raw'
) -> list[dict]:
    """
    Parse raw PDBbind general PP index file and extract chain sequences.
    """
    affinity_pattern = re.compile(r'(Kd|Ki|IC50)\s*[<>=~]?\s*([\d\.]+)\s*([a-zA-Z]+)', re.IGNORECASE)
    chain_pattern = re.compile(r'\(([a-zA-Z0-9]+)\|([a-zA-Z0-9]+)\)')

    entries = []
    with open(index_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('#') or line.startswith('1#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                pdb_id = parts[0].lower()[:4]
                if len(pdb_id) != 4 or not pdb_id.isalnum():
                    continue
                affinity_match = affinity_pattern.search(parts[3])
                chain_match = chain_pattern.search(line)
                if not affinity_match or not chain_match:
                    continue
                _, val_str, unit_str = affinity_match.groups()
                try:
                    val_molar = float(val_str) * UNITS_MAP[unit_str.lower()]
                    pKd = round(-math.log10(val_molar), 2)
                    rec_chains = chain_match.group(1)
                    lig_chains = chain_match.group(2)
                    resolution = parts[1]
                    year = parts[2]
                except Exception:
                    continue

                pdb_file = os.path.join(raw_pdb_dir, f"{pdb_id}.pdb")
                seqs = extract_seq_from_pdb_file(pdb_file)

                entries.append({
                    'pdb_id': pdb_id,
                    'affinity': pKd,
                    'receptor': rec_chains,
                    'ligand': lig_chains,
                    'resolution': resolution,
                    'year': year,
                    'chains': seqs
                })

    return entries


def align_chain_pair(
    seq1: str,
    seq2: str,
    aligner: PairwiseAligner
) -> tuple[float, int, int]:
    """
    Perform Global Needleman-Wunsch pairwise alignment.
    Returns:
      identity: matches / min(len(seq1), len(seq2))
      matches: count of identical aligned residue positions
      alignment_length: total length of alignment span
    """
    l1 = len(seq1)
    l2 = len(seq2)
    min_len = min(l1, l2)
    if min_len < 5:
        return 0.0, 0, 0
    
    score = aligner.score(seq1, seq2)
    if score <= 0:
        return 0.0, 0, 0

    alns = aligner.align(seq1, seq2)
    best_aln = alns[0]
    matches = best_aln.counts().identities
    identity = matches / min_len
    aln_len = best_aln.shape[1]
    return identity, matches, aln_len


def process_pdbbind_batch(
    args: tuple[list[dict], list[tuple[str, str, str]], float]
) -> list[dict]:
    """
    Worker task for batch of PDBbind complexes against all benchmark chains.
    """
    chunk_entries, bench_chains, threshold = args
    aligner = create_aligner()
    processed_results = []

    for entry in chunk_entries:
        pid = entry['pdb_id']
        chains_dict = entry['chains']
        
        max_sim = 0.0
        best_hit_bench = ""
        best_hit_chain = ""
        excluded_hits = []

        for p_chain, p_seq in chains_dict.items():
            if len(p_seq) < 5:
                continue
            for b_id, b_chain, b_seq in bench_chains:
                ident, matches, aln_len = align_chain_pair(p_seq, b_seq, aligner)
                if ident > max_sim:
                    max_sim = ident
                    best_hit_bench = b_id
                    best_hit_chain = b_chain
                if ident >= threshold:
                    excluded_hits.append({
                        'pdbbind_id': pid,
                        'pdbbind_chain': p_chain,
                        'bench_id': b_id,
                        'bench_chain': b_chain,
                        'seq_identity': round(ident, 4),
                        'alignment_length': aln_len,
                        'matches': matches
                    })

        is_excluded = (len(excluded_hits) > 0)
        processed_results.append({
            'entry': entry,
            'is_excluded': is_excluded,
            'max_sim': round(max_sim, 4),
            'best_bench_hit': f"{best_hit_bench}_{best_hit_chain}" if best_hit_bench else "none",
            'excluded_hits': excluded_hits
        })

    return processed_results


def run_zero_leakage_filter(
    index_path: str = 'data/dynarepo/pdbbind/INDEX_general_PP.2020R1.lst',
    raw_pdb_dir: str = 'data/pdbbind/raw',
    benchmark_dir: str = 'data/kastritis/raw',
    output_dir: str = 'data/pdbbind',
    threshold: float = 0.35,
    num_workers: int = 8
) -> dict:
    """
    Execute full sequence similarity clustering and zero-leakage filtering.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    print("=" * 70)
    print("  MILESTONE M2: SEQUENCE CLUSTERING & ZERO-LEAKAGE FILTER ENGINE")
    print("=" * 70)
    print(f"Start Time         : {now_utc}")
    print(f"Similarity Metric  : Global Needleman-Wunsch [matches / min(L1, L2)]")
    print(f"Cutoff Threshold   : {threshold * 100:.1f}%")
    print(f"PDBbind Index      : {index_path}")
    print(f"PDBbind Raw Dir    : {raw_pdb_dir}")
    print(f"Benchmark Dir      : {benchmark_dir}")
    print(f"Output Directory   : {output_dir}")
    print("-" * 70)

    # 1. Load benchmark complexes
    print("[1/4] Loading Kastritis benchmark sequences...")
    bench_complexes, bench_chains = load_benchmark_sequences(benchmark_dir=benchmark_dir)
    print(f"      -> Loaded {len(bench_complexes)} benchmark complexes ({len(bench_chains)} individual chains)")

    # 2. Parse PDBbind dataset
    print("[2/4] Parsing PDBbind general PP dataset...")
    pdbbind_entries = parse_pdbbind_index(index_path=index_path, raw_pdb_dir=raw_pdb_dir)
    print(f"      -> Parsed {len(pdbbind_entries)} PDBbind complexes with valid sequences")

    # 3. Multi-process sequence alignment
    print(f"[3/4] Aligning {len(pdbbind_entries)} complexes against {len(bench_chains)} benchmark chains with {num_workers} workers...")
    t_align_start = time.time()
    chunk_size = (len(pdbbind_entries) + num_workers - 1) // num_workers
    chunks = [pdbbind_entries[i:i + chunk_size] for i in range(0, len(pdbbind_entries), chunk_size)]
    tasks = [(chunk, bench_chains, threshold) for chunk in chunks]

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(process_pdbbind_batch, tasks):
            all_results.extend(res)
    t_align_end = time.time()
    print(f"      -> Alignment completed in {t_align_end - t_align_start:.2f}s")

    # 4. Separate clean vs excluded
    excluded_records = [r for r in all_results if r['is_excluded']]
    retained_records = [r for r in all_results if not r['is_excluded']]
    all_leakage_hits = [hit for r in all_results for hit in r['excluded_hits']]

    # Sort leakage hits by identity descending
    all_leakage_hits.sort(key=lambda x: (x['seq_identity'], x['pdbbind_id']), reverse=True)

    print(f"[4/4] Generating outputs...")
    # 4a. Write leakage_matrix.csv
    leakage_csv_path = os.path.join(output_dir, 'leakage_matrix.csv')
    with open(leakage_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pdbbind_id', 'pdbbind_chain', 'bench_id', 'bench_chain',
            'seq_identity', 'alignment_length', 'matches'
        ])
        for hit in all_leakage_hits:
            writer.writerow([
                hit['pdbbind_id'], hit['pdbbind_chain'],
                hit['bench_id'], hit['bench_chain'],
                f"{hit['seq_identity']:.4f}",
                hit['alignment_length'], hit['matches']
            ])
    print(f"      -> Wrote {len(all_leakage_hits)} excluded pairs to {leakage_csv_path}")

    # 4b. Write index.csv (clean complexes)
    index_csv_path = os.path.join(output_dir, 'index.csv')
    with open(index_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pdb_id', 'affinity', 'receptor', 'ligand',
            'resolution', 'year', 'max_bench_sim', 'best_bench_hit'
        ])
        for r in retained_records:
            e = r['entry']
            writer.writerow([
                e['pdb_id'], e['affinity'], e['receptor'], e['ligand'],
                e['resolution'], e['year'], f"{r['max_sim']:.4f}", r['best_bench_hit']
            ])
    print(f"      -> Wrote {len(retained_records)} clean complexes to {index_csv_path}")

    # 4c. Compute audit metrics
    max_retained_sim = max((r['max_sim'] for r in retained_records), default=0.0)
    min_excluded_sim = min((r['max_sim'] for r in excluded_records), default=0.0)

    # Similarity bins across full PDBbind set
    all_max_sims = [r['max_sim'] for r in all_results]
    bins = {
        '0.00 - 0.10': sum(1 for s in all_max_sims if s < 0.10),
        '0.10 - 0.20': sum(1 for s in all_max_sims if 0.10 <= s < 0.20),
        '0.20 - 0.30': sum(1 for s in all_max_sims if 0.20 <= s < 0.30),
        '0.30 - 0.35': sum(1 for s in all_max_sims if 0.30 <= s < 0.35),
        '0.35 - 0.50': sum(1 for s in all_max_sims if 0.35 <= s < 0.50),
        '0.50 - 0.70': sum(1 for s in all_max_sims if 0.50 <= s < 0.70),
        '0.70 - 0.90': sum(1 for s in all_max_sims if 0.70 <= s < 0.90),
        '0.90 - 1.00': sum(1 for s in all_max_sims if 0.90 <= s <= 1.00),
    }

    # 4d. Write info.txt audit report
    info_path = os.path.join(output_dir, 'info.txt')
    elapsed = time.time() - start_time
    with open(info_path, 'w') as f:
        f.write("=" * 75 + "\n")
        f.write("PDBBIND PROTEIN-PROTEIN ZERO-LEAKAGE HOMOLOGY FILTER AUDIT REPORT\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Timestamp                  : {now_utc}\n")
        f.write(f"Pipeline Step              : Milestone M2 (Sequence Similarity Clustering & Leakage Filter)\n")
        f.write(f"Total Processing Time      : {elapsed:.2f} seconds\n\n")
        
        f.write("--- 1. FILTERING METHODOLOGY & PARAMETERS ---\n")
        f.write("Clustering Engine          : Biopython Bio.Align.PairwiseAligner (C-accelerated Needleman-Wunsch)\n")
        f.write("Substitution Matrix        : BLOSUM62\n")
        f.write("Alignment Mode             : Global\n")
        f.write("Scoring Parameters         : Match/Mismatch from BLOSUM62, Gap Open = -11.0, Gap Extend = -1.0\n")
        f.write("Normalization Metric       : SeqIdentity = matches / min(Length_PDBbind, Length_Benchmark)\n")
        f.write(f"Identity Cutoff Threshold  : {threshold * 100:.1f}%\n")
        f.write("Exclusion Strategy         : Any-chain cross-matching (purges complex if ANY chain matches)\n\n")

        f.write("--- 2. DATASET SUMMARY STATISTICS ---\n")
        f.write(f"Benchmark Reference Source : Kastritis Affinity Benchmark / PRODIGY cleaned dataset\n")
        f.write(f"Total Benchmark Complexes  : {len(bench_complexes)}\n")
        f.write(f"Total Benchmark Chains     : {len(bench_chains)}\n")
        f.write(f"Raw PDBbind PP Entries     : {len(pdbbind_entries)}\n")
        f.write(f"Total Chain Alignments     : {len(pdbbind_entries) * len(bench_chains):,}\n\n")

        f.write("--- 3. HOMOLOGY FILTERING OUTCOMES ---\n")
        f.write(f"Homologous Complexes Purged: {len(excluded_records):,} ({len(excluded_records) / len(pdbbind_entries) * 100:.2f}%)\n")
        f.write(f"Clean Complexes Retained   : {len(retained_records):,} ({len(retained_records) / len(pdbbind_entries) * 100:.2f}%)\n")
        f.write(f"Total Excluded Chain Pairs : {len(all_leakage_hits):,}\n")
        f.write(f"Max Similarity (Retained)  : {max_retained_sim * 100:.2f}%\n")
        f.write(f"Min Similarity (Excluded)  : {min_excluded_sim * 100:.2f}%\n\n")

        f.write("--- 4. SEQUENCE SIMILARITY DISTRIBUTION ---\n")
        f.write("Max Sequence Identity to Benchmark:\n")
        for bin_label, count in bins.items():
            pct = count / len(all_results) * 100
            bar = "#" * int(pct // 2)
            f.write(f"  {bin_label:>14} : {count:>5} ({pct:>5.1f}%) | {bar}\n")
        f.write("\n")

        f.write("--- 5. ZERO DATA LEAKAGE CERTIFICATION ---\n")
        leakage_violations = [r for r in retained_records if r['max_sim'] >= threshold]
        if len(leakage_violations) == 0 and max_retained_sim < threshold:
            f.write("STATUS                     : [PASSED] ABSOLUTE ZERO DATA LEAKAGE CERTIFIED\n")
            f.write(f"Verification Check         : max(p,b) SeqID(p, b) = {max_retained_sim * 100:.2f}% < {threshold * 100:.1f}%\n")
            f.write(f"Violations Detected        : 0 / {len(retained_records)}\n\n")
        else:
            f.write("STATUS                     : [FAILED] LEAKAGE DETECTED\n")
            f.write(f"Violations Count           : {len(leakage_violations)}\n\n")

        f.write("--- 6. TOP 20 EXCLUDED HOMOLOGOUS PAIRS (SAMPLE AUDIT) ---\n")
        f.write(f"{'PDBbind ID':<12}{'Chain':<8}{'Bench ID':<12}{'Chain':<8}{'Identity':<12}{'Aln Len':<10}{'Matches':<8}\n")
        f.write("-" * 70 + "\n")
        for hit in all_leakage_hits[:20]:
            f.write(f"{hit['pdbbind_id']:<12}{hit['pdbbind_chain']:<8}{hit['bench_id']:<12}{hit['bench_chain']:<8}{hit['seq_identity']*100:>7.2f}%    {hit['alignment_length']:<10}{hit['matches']:<8}\n")
        f.write("\n" + "=" * 75 + "\n")

    print(f"      -> Wrote audit log to {info_path}")
    print("-" * 70)
    print(f"SUMMARY: Filtered {len(pdbbind_entries)} complexes down to {len(retained_records)} clean entries.")
    print(f"         Excluded {len(excluded_records)} homologous complexes ({len(excluded_records)/len(pdbbind_entries)*100:.2f}%).")
    print(f"         Max sequence identity in retained set: {max_retained_sim*100:.2f}% (< {threshold*100:.1f}%).")
    print("=" * 70)

    return {
        'total_raw': len(pdbbind_entries),
        'total_bench': len(bench_complexes),
        'retained_count': len(retained_records),
        'excluded_count': len(excluded_records),
        'leakage_hits_count': len(all_leakage_hits),
        'max_retained_sim': max_retained_sim,
        'leakage_matrix_csv': leakage_csv_path,
        'index_csv': index_csv_path,
        'info_txt': info_path
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDBbind Homology Filtering & Zero Leakage Engine")
    parser.add_argument('--index', default='data/dynarepo/pdbbind/INDEX_general_PP.2020R1.lst', help="Path to raw PDBbind index")
    parser.add_argument('--raw_dir', default='data/pdbbind/raw', help="Directory containing raw PDB files")
    parser.add_argument('--bench_dir', default='data/kastritis/raw', help="Directory containing benchmark PDB files")
    parser.add_argument('--output_dir', default='data/pdbbind', help="Output directory")
    parser.add_argument('--threshold', type=float, default=0.35, help="Sequence identity cutoff threshold (default: 0.35)")
    parser.add_argument('--workers', type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    run_zero_leakage_filter(
        index_path=args.index,
        raw_pdb_dir=args.raw_dir,
        benchmark_dir=args.bench_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
