import urllib.request
import urllib.parse
import time
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

CLASSES = [
    (
        "Enzyme",
        'reviewed:true AND keyword:"Hydrolase [KW-0378]"',
        500,
    ),
    (
        "Transporter",
        'reviewed:true AND keyword:"Transporter [KW-0813]"',
        500,
    ),
    (
        "Transcription_Factor",
        'reviewed:true AND keyword:"Transcription regulation [KW-0804]" NOT keyword:"Hydrolase [KW-0378]"',
        500,
    ),
    (
        "Structural_Protein",
        'reviewed:true AND keyword:"Extracellular matrix [KW-0胞261]" OR (reviewed:true AND keyword:"Structural protein [KW-0261]")',
        500,
    ),
]

CLASSES = [
    ("Enzyme",               "reviewed:true AND keyword:KW-0378", 500),
    ("Transporter",          "reviewed:true AND keyword:KW-0813", 500),
    ("Transcription_Factor", "reviewed:true AND keyword:KW-0804", 500),
    ("Structural_Protein",   "reviewed:true AND keyword:KW-0261", 500),
]

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"


def download_fasta(query: str, size: int, out_path: str) -> int:
    """Download `size` FASTA sequences from UniProt and save to out_path."""
    params = urllib.parse.urlencode({
        "query":  query,
        "format": "fasta",
        "size":   size,
    })
    url = f"{BASE_URL}?{params}"
    print(f"  GET {url[:120]}...")

    req = urllib.request.Request(url, headers={"User-Agent": "ProteinClassifier/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read().decode("utf-8")

    if not data.strip():
        raise ValueError("Empty response from UniProt")

    n_seqs = data.count(">")
    with open(out_path, "w") as f:
        f.write(data)

    print(f"  ✓ Saved {n_seqs} sequences → {out_path}")
    return n_seqs


def main():
    print("=" * 60)
    print("Downloading UniProt Swiss-Prot sequences")
    print("=" * 60)

    for label, query, size in CLASSES:
        out_path = os.path.join(RAW_DIR, f"{label}.fasta")

        if os.path.exists(out_path):
            n = open(out_path).read().count(">")
            print(f"  [SKIP] {label} already downloaded ({n} seqs)")
            continue

        print(f"\n[{label}]")
        try:
            download_fasta(query, size, out_path)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            sys.exit(1)

        time.sleep(1)  

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
