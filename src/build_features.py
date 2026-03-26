import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import extract_features

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CLASSES = [
    "Enzyme",
    "Transporter",
    "Transcription_Factor",
    "Structural_Protein",
]


def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) tuples."""
    records = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            elif line:
                seq_parts.append(line)
    if header:
        records.append((header, "".join(seq_parts)))
    return records


def extract_uniprot_id(header: str) -> str:
    """Extract UniProt accession from FASTA header line."""
    m = re.search(r'\|([A-Z0-9]+)\|', header)
    return m.group(1) if m else header.split()[0]


def build_feature_matrix() -> pd.DataFrame:
    rows = []
    print("=" * 60)
    print("Building feature matrix from FASTA files")
    print("=" * 60)

    for label in CLASSES:
        fasta_path = os.path.join(RAW_DIR, f"{label}.fasta")
        if not os.path.exists(fasta_path):
            print(f"  ✗ Missing: {fasta_path}")
            print("    Run src/download_data.py first.")
            sys.exit(1)

        records = parse_fasta(fasta_path)
        print(f"\n[{label}] {len(records)} sequences loaded")

        n_ok, n_skip = 0, 0
        for header, seq in records:
            feats = extract_features(seq)
            if feats is None:
                n_skip += 1
                continue
            feats["entry_id"] = extract_uniprot_id(header)
            feats["label"]    = label
            rows.append(feats)
            n_ok += 1

        print(f"  ✓ {n_ok} extracted  |  {n_skip} skipped (too short / invalid)")

    df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "features.csv")
    df.to_csv(out_path, index=False)

    print(f"\nFeature matrix: {df.shape[0]} proteins × {df.shape[1]-2} features")
    print(f"   Saved → {out_path}")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    build_feature_matrix()
