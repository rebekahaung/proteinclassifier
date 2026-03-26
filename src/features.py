import re
import numpy as np

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBICITY = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

AA_WEIGHTS = {
    'A':  89.04768, 'R': 174.11168, 'N': 132.05349, 'D': 133.03751,
    'C': 121.01975, 'Q': 146.06914, 'E': 147.05316, 'G':  75.03203,
    'H': 155.06948, 'I': 131.09463, 'L': 131.09463, 'K': 146.10553,
    'M': 149.05105, 'F': 165.07898, 'P': 115.06333, 'S': 105.04259,
    'T': 119.05824, 'W': 204.08988, 'Y': 181.07389, 'V': 117.07898,
}
WATER = 18.01056

PKA = {
    'D': 3.9, 'E': 4.07, 'H': 6.04, 'C': 8.18,
    'Y': 10.46, 'K': 10.53, 'R': 12.48,
    'Nterm': 8.0, 'Cterm': 3.1,
}

EXTINCTION_COEFF = {'W': 5500, 'Y': 1490, 'C': 125}

HELIX_PROP  = {'A':1.45,'R':0.98,'N':0.73,'D':0.98,'C':0.77,'Q':1.17,'E':1.53,
               'G':0.53,'H':1.24,'I':1.00,'L':1.34,'K':1.07,'M':1.20,'F':1.12,
               'P':0.59,'S':0.79,'T':0.82,'W':1.14,'Y':0.61,'V':1.14}
TURN_PROP   = {'A':0.77,'R':0.88,'N':1.41,'D':1.41,'C':0.81,'Q':0.98,'E':0.99,
               'G':1.64,'H':0.68,'I':0.51,'L':0.58,'K':0.96,'M':0.48,'F':0.59,
               'P':1.91,'S':1.43,'T':1.04,'W':0.76,'Y':1.05,'V':0.82}
SHEET_PROP  = {'A':0.97,'R':0.93,'N':0.65,'D':0.72,'C':1.30,'Q':1.23,'E':0.26,
               'G':0.81,'H':0.71,'I':1.60,'L':1.22,'K':0.74,'M':1.67,'F':1.28,
               'P':0.62,'S':0.72,'T':1.20,'W':1.19,'Y':1.29,'V':1.65}

DIPEPTIDES = ['KK','EE','RR','DD','FF','WW','PP','GG',
              'AA','LL','VV','II','SS','TT']


def clean(seq: str) -> str:
    """Strip non-standard amino acids and uppercase."""
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper().strip())



def aa_composition(seq: str) -> dict:
    n = len(seq)
    return {f'aa_{a}': seq.count(a) / n for a in AA_LIST}


def molecular_weight(seq: str) -> float:
    return sum(AA_WEIGHTS.get(a, 0) for a in seq) - WATER * (len(seq) - 1)


def gravy(seq: str) -> float:
    """Grand Average of Hydropathicity (Kyte-Doolittle)."""
    return sum(HYDROPHOBICITY.get(a, 0) for a in seq) / len(seq)


def aromaticity(seq: str) -> float:
    return (seq.count('F') + seq.count('W') + seq.count('Y')) / len(seq)


def instability_index(seq: str) -> float:
    """
    Guruprasad et al. (1990) instability index.
    Proteins with II < 40 are considered stable.
    """
    DIWV = {
        ('W','W'):1.0,('W','C'):1.0,('W','M'):24.68,('W','H'):24.68,
        ('C','K'):1.0,('C','C'):1.0,('A','W'):1.0,
        ('D','W'):6.98,('E','W'):6.98,
        ('N','W'):13.34,('N','F'):-14.03,
        ('G','W'):13.34,('Q','W'):1.0,
        ('Y','K'):4.96,('Y','W'):1.0,
        ('K','K'):1.0,('F','K'):1.0,
        ('H','E'):1.0,('R','W'):1.0,
        ('S','W'):1.0,('T','W'):1.0,
        ('V','W'):1.0,('M','W'):1.0,
        ('P','W'):1.0,('P','K'):1.0,
        ('I','W'):1.0,('L','W'):1.0,
    }
    if len(seq) < 2:
        return 0.0
    total = sum(DIWV.get((seq[i], seq[i+1]), 1.0) for i in range(len(seq)-1))
    return (10.0 / len(seq)) * total


def isoelectric_point(seq: str) -> float:
    """Estimate pI via bisection method."""
    counts = {r: seq.count(r) for r in 'DEHCYKR'}

    def charge(ph):
        c  =  1.0 / (1 + 10 ** (ph - PKA['Nterm']))   # N-term
        c -= 1.0 / (1 + 10 ** (PKA['Cterm'] - ph))    # C-term
        for r in ('D', 'E'):
            c -= counts[r] / (1 + 10 ** (PKA[r] - ph))
        for r in ('H', 'K', 'R'):
            c += counts[r] / (1 + 10 ** (ph - PKA[r]))
        for r in ('C', 'Y'):
            c -= counts[r] / (1 + 10 ** (PKA[r] - ph))
        return c

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def secondary_structure(seq: str) -> tuple:
    """Chou-Fasman estimated helix/turn/sheet fractions."""
    h = np.mean([HELIX_PROP.get(a, 1.0) for a in seq])
    t = np.mean([TURN_PROP.get(a, 1.0)  for a in seq])
    s = np.mean([SHEET_PROP.get(a, 1.0) for a in seq])
    total = h + t + s
    return h/total, t/total, s/total


def molar_extinction(seq: str) -> float:
    return sum(seq.count(a) * v for a, v in EXTINCTION_COEFF.items())


def charge_ph7(seq: str) -> float:
    return (seq.count('K') + seq.count('R') + seq.count('H') * 0.1
            - seq.count('D') - seq.count('E'))


def dipeptide_composition(seq: str) -> dict:
    n = max(len(seq) - 1, 1)
    return {f'dp_{p}': seq.count(p) / n for p in DIPEPTIDES}



def extract_features(seq: str) -> dict | None:
    """
    Extract all 45 features from a raw protein sequence string.
    Returns None if the sequence is too short or invalid.
    """
    seq = clean(seq)
    if len(seq) < 10:
        return None

    feats = {}
    feats.update(aa_composition(seq))

    feats['molecular_weight']  = molecular_weight(seq)
    feats['gravy']             = gravy(seq)
    feats['aromaticity']       = aromaticity(seq)
    feats['instability_index'] = instability_index(seq)
    feats['isoelectric_point'] = isoelectric_point(seq)
    feats['molar_extinction']  = molar_extinction(seq)
    feats['charge_ph7']        = charge_ph7(seq)
    feats['length']            = len(seq)

    h, t, s = secondary_structure(seq)
    feats['helix_fraction'] = h
    feats['turn_fraction']  = t
    feats['sheet_fraction'] = s

    feats.update(dipeptide_composition(seq))

    return feats


# List feature names in extraction order
def feature_names() -> list[str]:
    dummy = extract_features("ACDEFGHIKLMNPQRSTVWY" * 5)
    return list(dummy.keys())

FEATURE_NAMES = feature_names()
