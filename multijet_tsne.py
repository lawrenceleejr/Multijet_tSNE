#!/usr/bin/env python3
"""
multijet_tsne.py — t-SNE analysis of multijet LHC events for pair-produced resonances.

For each event, all ways to split jets into two triplets are enumerated.
Features of each splitting are computed and fed into t-SNE.
The plot highlights the truth-correct grouping (from parent_pdg labels) in red.

Usage (via Docker — see run.sh):
    ./run.sh ./data/events.h5 [options]

Direct usage:
    python multijet_tsne.py events.h5 [more.h5 ...] [options]

Options:
    --max-events N    Maximum events to process (default: all)
    --perplexity F    t-SNE perplexity (default: 30)
    --output FILE     Output plot path (default: tsne_multijet.png)
    --no-normalize    Skip StandardScaler on features
    --seed INT        Random seed (default: 42)
    --n-iter INT      t-SNE iterations (default: 1000)
"""

import argparse
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — required inside Docker
import matplotlib.pyplot as plt
import numpy as np

# ─── Column indices into jet_features ─────────────────────────────────────────

JET_PT    = 0
JET_ETA   = 1
JET_PHI   = 2
JET_MASS  = 3
JET_PDG   = 5   # parent_pdg — identifies which resonance the jet came from

# ─── Labels ───────────────────────────────────────────────────────────────────

LABEL_WRONG   =  0
LABEL_CORRECT =  1
LABEL_AMBIG   = -1

FEATURE_NAMES = [
    "m_min", "m_max", "m_avg", "m_asym",
    "pt_min", "pt_max", "pt_avg", "pt_asym",
    "dR_between",
    "dR_A_min", "dR_A_max", "dR_A_mean",
    "dR_B_min", "dR_B_max", "dR_B_mean",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Physics primitives
# ═══════════════════════════════════════════════════════════════════════════════

def jets_to_4vectors(jets: np.ndarray) -> np.ndarray:
    """
    Convert (pt, eta, phi, mass) to (E, px, py, pz).

    Parameters
    ----------
    jets : (n, 7) array  — columns: [pt, eta, phi, mass, n_const, pdg, is_sig]

    Returns
    -------
    vecs : (n, 4) array  — columns: [E, px, py, pz]
    """
    pt   = jets[:, JET_PT]
    eta  = jets[:, JET_ETA]
    phi  = jets[:, JET_PHI]
    mass = jets[:, JET_MASS]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E  = np.sqrt(np.maximum(0.0, pt**2 * np.cosh(eta)**2 + mass**2))

    return np.stack([E, px, py, pz], axis=1)


def invariant_mass(vec4: np.ndarray) -> float:
    """Invariant mass of a summed 4-vector [E, px, py, pz]."""
    m2 = vec4[0]**2 - vec4[1]**2 - vec4[2]**2 - vec4[3]**2
    return float(np.sqrt(max(0.0, m2)))


def triplet_pt_eta_phi(vec4: np.ndarray):
    """Extract (pt, eta, phi) of a composite 4-vector [E, px, py, pz]."""
    px, py, pz = vec4[1], vec4[2], vec4[3]
    pt = np.sqrt(px**2 + py**2)
    if pt < 1e-9:
        eta = 0.0
    else:
        p_tot = np.sqrt(pt**2 + pz**2)
        eta = float(np.arctanh(np.clip(pz / (p_tot + 1e-12), -0.9999, 0.9999)))
    phi = float(np.arctan2(py, px))
    return float(pt), eta, phi


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """ΔR = sqrt(Δη² + Δφ²), with φ wrapped to (−π, π]."""
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt((eta1 - eta2)**2 + dphi**2))


def pairwise_delta_r(jets: np.ndarray) -> list:
    """
    All pairwise ΔR values for a set of jets.

    Parameters
    ----------
    jets : (n, 7) array

    Returns
    -------
    list of float, length C(n, 2)
    """
    etas = jets[:, JET_ETA]
    phis = jets[:, JET_PHI]
    result = []
    for i, j in combinations(range(len(jets)), 2):
        result.append(delta_r(etas[i], phis[i], etas[j], phis[j]))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Combinatorics
# ═══════════════════════════════════════════════════════════════════════════════

def get_splittings(n_jets: int) -> list:
    """
    Return all unique unordered pairs of disjoint triplets from n_jets jets.

    Canonical form: jet 0 is always in group A — this halves the count and
    avoids (A, B) and (B, A) being treated as different splittings.

    For n_jets=6: returns 10 splittings.
    For n_jets=7: jet 0 is fixed in A; choose 2 more from {1..6} (C(6,2)=15);
                  group B is any 3 of the remaining 4 jets (C(4,3)=4) → 60 splittings.

    Returns
    -------
    list of (tuple_A, tuple_B) where each is a sorted tuple of jet indices
    """
    indices = list(range(n_jets))
    remaining = indices[1:]  # jet 0 always in A

    splittings = []
    for extra_a in combinations(remaining, 2):
        group_a = tuple(sorted((0,) + extra_a))
        rest = [i for i in remaining if i not in extra_a]
        for group_b in combinations(rest, 3):
            group_b = tuple(sorted(group_b))
            splittings.append((group_a, group_b))

    return splittings


# ═══════════════════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_splitting_features(
    jets: np.ndarray,
    vecs4: np.ndarray,
    group_a: tuple,
    group_b: tuple,
) -> np.ndarray:
    """
    Compute the 15-feature vector for one (3+3) splitting.

    Parameters
    ----------
    jets  : (n, 7) full jet array for the event
    vecs4 : (n, 4) precomputed 4-vectors [E, px, py, pz]
    group_a, group_b : tuples of jet indices (length 3 each)

    Returns
    -------
    features : (15,) float32 array
    """
    list_a, list_b = list(group_a), list(group_b)

    # Composite 4-vectors
    sum_a = vecs4[list_a].sum(axis=0)
    sum_b = vecs4[list_b].sum(axis=0)

    m_a = invariant_mass(sum_a)
    m_b = invariant_mass(sum_b)
    pt_a, eta_a, phi_a = triplet_pt_eta_phi(sum_a)
    pt_b, eta_b, phi_b = triplet_pt_eta_phi(sum_b)

    # Pairwise ΔR within each triplet
    dR_a = pairwise_delta_r(jets[list_a])  # 3 values
    dR_b = pairwise_delta_r(jets[list_b])  # 3 values

    # Canonical ordering: triplet with smaller min-ΔR gets label 'A'
    # (pure symmetrisation — prevents the same splitting mapping to two
    #  different feature vectors depending on label assignment)
    if min(dR_a) > min(dR_b):
        m_a, m_b     = m_b, m_a
        pt_a, pt_b   = pt_b, pt_a
        eta_a, eta_b = eta_b, eta_a
        phi_a, phi_b = phi_b, phi_a
        dR_a, dR_b   = dR_b, dR_a

    m_min  = min(m_a,  m_b)
    m_max  = max(m_a,  m_b)
    m_avg  = (m_a + m_b) / 2.0
    m_asym = (m_max - m_min) / (m_max + m_min + 1e-9)

    pt_min  = min(pt_a,  pt_b)
    pt_max  = max(pt_a,  pt_b)
    pt_avg  = (pt_a + pt_b) / 2.0
    pt_asym = (pt_max - pt_min) / (pt_max + pt_min + 1e-9)

    dR_between = delta_r(eta_a, phi_a, eta_b, phi_b)

    return np.array([
        m_min, m_max, m_avg, m_asym,
        pt_min, pt_max, pt_avg, pt_asym,
        dR_between,
        min(dR_a), max(dR_a), float(np.mean(dR_a)),
        min(dR_b), max(dR_b), float(np.mean(dR_b)),
    ], dtype=np.float32)


def identify_correct_splitting(jets: np.ndarray, splittings: list):
    """
    Identify the index of the truth-correct splitting using parent_pdg labels.

    Works when the two resonances have *different* non-zero parent_pdg values
    (e.g. squark / anti-squark: +1000006 / -1000006).

    Returns None (ambiguous) when:
      - all parent_pdg are 0 (no truth info stored)
      - only 1 distinct non-zero PDG (self-conjugate like gluinos: both = 1000021)
      - the two PDGs don't each cover exactly n//2 jets
      - the resulting groups aren't found in the splittings list

    Parameters
    ----------
    jets       : (n, 7) array for the event (already selected/masked)
    splittings : list of (group_a, group_b) tuples as returned by get_splittings

    Returns
    -------
    int or None
    """
    pdgs = jets[:, JET_PDG].astype(int)
    half = len(jets) // 2

    nonzero = pdgs[pdgs != 0]
    if len(nonzero) == 0:
        return None  # no truth info at all

    counts = Counter(nonzero.tolist())
    unique = list(counts.keys())

    # Need exactly 2 distinct non-zero PDG values (particle / anti-particle)
    if len(unique) != 2:
        return None

    # Each PDG must cover exactly half the jets (all jets must be signal jets)
    if counts[unique[0]] != half or counts[unique[1]] != half:
        return None

    # Build the canonical groups
    idx_p0 = tuple(sorted(int(i) for i, p in enumerate(pdgs) if p == unique[0]))
    idx_p1 = tuple(sorted(int(i) for i, p in enumerate(pdgs) if p == unique[1]))
    correct = (idx_p0, idx_p1) if idx_p0 < idx_p1 else (idx_p1, idx_p0)

    try:
        return splittings.index(correct)
    except ValueError:
        return None


def identify_correct_splitting_from_truth(
    splittings: list,
    truth_group_a: tuple,
    truth_group_b: tuple,
) -> int | None:
    """
    Identify the correct splitting index using explicit truth jet groups.

    Parameters
    ----------
    splittings     : list of (group_a, group_b) tuples (sorted, canonical form)
    truth_group_a  : sorted tuple of jet indices for the first decay chain
    truth_group_b  : sorted tuple of jet indices for the second decay chain

    Both groups are in the *post-sort* (pT-ordered) coordinate system.

    Returns
    -------
    int or None (if the truth groups don't appear in the splittings list)
    """
    # Canonical form: the group containing sorted index 0 must be group_a
    if 0 in truth_group_b:
        truth_group_a, truth_group_b = truth_group_b, truth_group_a
    candidate = (truth_group_a, truth_group_b)
    try:
        return splittings.index(candidate)
    except ValueError:
        return None


def label_by_min_mass_asym(features: np.ndarray) -> int:
    """
    Heuristic label for self-conjugate resonances (e.g. gluinos) where
    parent_pdg cannot distinguish the two decay chains.

    For equal-mass pair production the correct 3+3 grouping should minimise
    the fractional mass asymmetry between the two triplets:
        m_asym = |m1 - m2| / (m1 + m2)

    Parameters
    ----------
    features : (n_splittings, 15) array for one event

    Returns
    -------
    int — index of the splitting with the smallest m_asym
    """
    return int(np.argmin(features[:, 3]))  # column 3 = m_asym


# ═══════════════════════════════════════════════════════════════════════════════
# Event processing
# ═══════════════════════════════════════════════════════════════════════════════

def select_valid_jets(jet_feats: np.ndarray, jet_mask: np.ndarray):
    """
    Apply the jet mask and return the valid jets sorted by pT (descending).

    Returns
    -------
    jets        : (n_valid, 7) array sorted by pT descending
    orig_indices: (n_valid,) int array of original (pre-sort) jet indices
    """
    valid_mask = np.asarray(jet_mask, dtype=bool)
    valid = jet_feats[valid_mask]
    orig_indices = np.where(valid_mask)[0]
    if len(valid) == 0:
        return valid, orig_indices
    order = np.argsort(valid[:, JET_PT])[::-1]
    return valid[order], orig_indices[order]


def process_event(jets: np.ndarray, min_jets: int = 6, truth_groups=None):
    """
    Process one event: enumerate all 3+3 splittings and compute features.

    Parameters
    ----------
    jets        : (n_valid, 7) sorted-by-pT jet array
    min_jets    : minimum number of jets required (default 6)
    truth_groups: optional (group_a, group_b) tuple of sorted jet-index tuples
                  in the pT-sorted coordinate system.  When provided it is used
                  directly instead of the PDG-based heuristic.

    Returns
    -------
    features : (n_splittings, 15) float32 array, or None if too few jets
    labels   : (n_splittings,) int array (LABEL_CORRECT/WRONG/AMBIG)
    """
    n = len(jets)
    if n < min_jets:
        return None, None

    splittings = get_splittings(n)
    n_splits   = len(splittings)

    vecs4    = jets_to_4vectors(jets)
    features = np.zeros((n_splits, len(FEATURE_NAMES)), dtype=np.float32)
    labels   = np.full(n_splits, LABEL_WRONG, dtype=np.int8)

    for k, (ga, gb) in enumerate(splittings):
        features[k] = compute_splitting_features(jets, vecs4, ga, gb)

    if truth_groups is not None:
        correct_idx = identify_correct_splitting_from_truth(
            splittings, truth_groups[0], truth_groups[1]
        )
    else:
        correct_idx = identify_correct_splitting(jets, splittings)

    if correct_idx is None:
        labels[:] = LABEL_AMBIG
    else:
        labels[correct_idx] = LABEL_CORRECT

    return features, labels


# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 loading
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_paths(raw_paths: list) -> list:
    """
    Accept a mix of HDF5 file paths and directories.
    Directories are globbed for *.h5 and *.hdf5 files.
    """
    resolved = []
    for p in raw_paths:
        path = Path(p)
        if path.is_dir():
            found = sorted(path.glob("*.h5")) + sorted(path.glob("*.hdf5"))
            if not found:
                print(f"[warn] No HDF5 files found in directory: {path}", file=sys.stderr)
            resolved.extend(found)
        elif path.exists():
            resolved.append(path)
        else:
            print(f"[warn] Path not found, skipping: {path}", file=sys.stderr)
    return resolved


def load_and_process(files: list, max_events: int = None, verbose: bool = False):
    """
    Load events from HDF5 files and compute features for all splittings.

    Returns
    -------
    all_features : (N_splittings_total, 15) float32
    all_labels   : (N_splittings_total,)   int8
    stats        : dict with summary counts
    """
    try:
        import h5py
    except ImportError:
        sys.exit("[error] h5py is required. Install via: pip install h5py")

    try:
        from tqdm import tqdm as _tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    all_features = []
    all_labels   = []

    n_events_seen    = 0
    n_events_used    = 0
    n_skipped_few    = 0
    n_skipped_nan    = 0
    n_ambiguous_evts = 0
    n_correct_evts   = 0

    for fpath in files:
        if verbose:
            print(f"[info] Loading {fpath}")
        try:
            with h5py.File(fpath, "r") as f:
                if "jet_features" not in f or "jet_mask" not in f:
                    print(f"[warn] Missing datasets in {fpath}, skipping", file=sys.stderr)
                    continue

                jet_feats_all = f["jet_features"][:]   # (N, max_jets, 7)
                jet_mask_all  = f["jet_mask"][:]        # (N, max_jets) bool

                # Read explicit truth jet-group assignments when available.
                # Expected layout: TARGETS/g1/{j1,j2,j3} and TARGETS/g2/{j1,j2,j3}
                # each containing per-event original jet indices.
                targets_g1 = None
                targets_g2 = None
                try:
                    if (
                        "TARGETS" in f
                        and "g1" in f["TARGETS"]
                        and "g2" in f["TARGETS"]
                        and all(k in f["TARGETS/g1"] for k in ("j1", "j2", "j3"))
                        and all(k in f["TARGETS/g2"] for k in ("j1", "j2", "j3"))
                    ):
                        targets_g1 = np.stack(
                            [f["TARGETS/g1/j1"][:], f["TARGETS/g1/j2"][:], f["TARGETS/g1/j3"][:]],
                            axis=1,
                        ).astype(int)   # (N, 3)
                        targets_g2 = np.stack(
                            [f["TARGETS/g2/j1"][:], f["TARGETS/g2/j2"][:], f["TARGETS/g2/j3"][:]],
                            axis=1,
                        ).astype(int)   # (N, 3)
                        if verbose:
                            print(f"[info]   Found TARGETS/g1,g2 truth labels")
                except Exception as exc:
                    print(f"[warn] Could not read TARGETS from {fpath}: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] Failed to read {fpath}: {exc}", file=sys.stderr)
            continue

        n_in_file = jet_feats_all.shape[0]
        iterator  = range(n_in_file)
        if use_tqdm:
            iterator = _tqdm(iterator, desc=f"  {Path(fpath).name}", unit="evt")

        for i in iterator:
            if max_events is not None and n_events_seen >= max_events:
                break
            n_events_seen += 1

            jets, orig_indices = select_valid_jets(jet_feats_all[i], jet_mask_all[i])

            if len(jets) < 6:
                n_skipped_few += 1
                continue

            # Guard against NaN/Inf in jet features
            if not np.all(np.isfinite(jets[:, :4])):
                n_skipped_nan += 1
                continue

            # Resolve truth groups into pT-sorted coordinates when available
            truth_groups = None
            if targets_g1 is not None and targets_g2 is not None:
                orig_to_sorted = {int(orig): pos for pos, orig in enumerate(orig_indices)}
                try:
                    ga = tuple(sorted(orig_to_sorted[idx] for idx in targets_g1[i]))
                    gb = tuple(sorted(orig_to_sorted[idx] for idx in targets_g2[i]))
                    if len(ga) == 3 and len(gb) == 3:
                        truth_groups = (ga, gb)
                except KeyError:
                    pass  # a truth jet was masked — fall back to PDG method

            feats, labels = process_event(jets, min_jets=6, truth_groups=truth_groups)
            if feats is None:
                n_skipped_few += 1
                continue

            all_features.append(feats)
            all_labels.append(labels)
            n_events_used += 1

            if LABEL_AMBIG in labels:
                n_ambiguous_evts += 1
            elif LABEL_CORRECT in labels:
                n_correct_evts += 1

        if max_events is not None and n_events_seen >= max_events:
            break

    stats = {
        "n_events_seen":    n_events_seen,
        "n_events_used":    n_events_used,
        "n_skipped_few":    n_skipped_few,
        "n_skipped_nan":    n_skipped_nan,
        "n_ambiguous_evts": n_ambiguous_evts,
        "n_correct_evts":   n_correct_evts,
    }

    if not all_features:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.int8), stats

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels,   axis=0),
        stats,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# t-SNE
# ═══════════════════════════════════════════════════════════════════════════════

def run_tsne(features: np.ndarray, perplexity: float, seed: int, n_iter: int,
             normalize: bool = True) -> np.ndarray:
    """
    Run t-SNE on the feature matrix.

    Parameters
    ----------
    features  : (N, 15) float32
    perplexity: t-SNE perplexity
    seed      : random seed
    n_iter    : number of optimisation iterations
    normalize : if True, apply StandardScaler first

    Returns
    -------
    embedding : (N, 2) float64
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X = features.copy()

    if normalize:
        X = StandardScaler().fit_transform(X)

    n = len(X)
    if n > 30_000:
        warnings.warn(
            f"Processing {n} splittings — t-SNE may be slow. "
            "Consider using --max-events to reduce the dataset."
        )

    # PCA init is more stable and faster than random
    # n_iter was renamed to max_iter in scikit-learn 1.2
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, n - 1),
        max_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        embedding = tsne.fit_transform(X)

    return embedding


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tsne(
    embedding: np.ndarray,
    labels: np.ndarray,
    stats: dict,
    output_path: str,
    perplexity: float,
):
    """
    Produce t-SNE scatter plot with correct/wrong/ambiguous coloring.

    Layers (back → front):
        1. Gray  — wrong interpretations
        2. Blue  — ambiguous (no truth label)
        3. Red   — correct interpretations
    """
    mask_wrong   = labels == LABEL_WRONG
    mask_correct = labels == LABEL_CORRECT
    mask_ambig   = labels == LABEL_AMBIG

    n_wrong   = int(mask_wrong.sum())
    n_correct = int(mask_correct.sum())
    n_ambig   = int(mask_ambig.sum())
    n_total   = len(labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Layer 1 — wrong
    if n_wrong > 0:
        ax.scatter(
            embedding[mask_wrong, 0], embedding[mask_wrong, 1],
            s=5, alpha=0.20, c="#888888", linewidths=0,
            label=f"Wrong ({n_wrong:,})",
            rasterized=True,
        )

    # Layer 2 — ambiguous
    if n_ambig > 0:
        ax.scatter(
            embedding[mask_ambig, 0], embedding[mask_ambig, 1],
            s=5, alpha=0.20, c="#457b9d", linewidths=0,
            label=f"Ambiguous / no truth ({n_ambig:,})",
            rasterized=True,
        )

    # Layer 3 — correct (on top, larger)
    if n_correct > 0:
        ax.scatter(
            embedding[mask_correct, 0], embedding[mask_correct, 1],
            s=35, alpha=0.85, c="#e63946",
            edgecolors="#9d0208", linewidths=0.4, zorder=5,
            label=f"Correct ({n_correct:,}, {100*n_correct/n_total:.1f}%)",
        )

    # Stats text box
    pct_correct = 100.0 * n_correct / n_total if n_total else 0.0
    stats_lines = [
        f"Events seen:        {stats['n_events_seen']:,}",
        f"Events processed:   {stats['n_events_used']:,}",
        f"Skipped (<6 jets):  {stats['n_skipped_few']:,}",
        f"Events w/ truth:    {stats['n_correct_evts']:,}",
        f"Events ambiguous:   {stats['n_ambiguous_evts']:,}",
        f"",
        f"Total splittings:   {n_total:,}",
        f"  Correct:  {n_correct:,}  ({pct_correct:.1f}%)",
        f"  Wrong:    {n_wrong:,}",
        f"  Ambig:    {n_ambig:,}",
    ]
    ax.text(
        0.02, 0.98, "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=7, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75, edgecolor="#cccccc"),
    )

    ax.set_xlabel("t-SNE component 1", fontsize=12)
    ax.set_ylabel("t-SNE component 2", fontsize=12)
    ax.set_title(
        f"t-SNE of multijet interpretations\n"
        f"perplexity={perplexity:.0f},  15 physics features per splitting",
        fontsize=13,
    )
    ax.legend(loc="upper right", markerscale=2.5, fontsize=9, framealpha=0.85)

    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n[done] Plot saved to: {output_path}")
    except IOError as exc:
        print(f"[error] Could not write plot: {exc}", file=sys.stderr)
        fallback = "/tmp/tsne_multijet.png"
        fig.savefig(fallback, dpi=150, bbox_inches="tight")
        print(f"[info] Saved to fallback path: {fallback}", file=sys.stderr)
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "t-SNE analysis of multijet LHC events to find pair-produced resonances.\n"
            "Enumerates all 3+3 jet splitting interpretations per event and plots\n"
            "the t-SNE embedding, highlighting the truth-correct grouping in red."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="+", metavar="path",
        help="HDF5 file(s) or directory containing *.h5 files",
    )
    parser.add_argument(
        "--max-events", type=int, default=None, metavar="N",
        help="Maximum number of events to process (default: all)",
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0, metavar="F",
        help="t-SNE perplexity (default: 30)",
    )
    parser.add_argument(
        "--output", default="tsne_multijet.png", metavar="FILE",
        help="Output plot path (default: tsne_multijet.png)",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip StandardScaler feature normalisation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for t-SNE (default: 42)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=1000,
        help="Number of t-SNE iterations (default: 1000)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-file loading progress",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve input paths
    files = resolve_paths(args.paths)
    if not files:
        sys.exit("[error] No HDF5 files found. Provide file paths or a directory.")

    print(f"[info] Found {len(files)} HDF5 file(s):")
    for f in files:
        print(f"         {f}")

    # Load and process
    print("\n[info] Processing events...")
    features, labels, stats = load_and_process(
        files,
        max_events=args.max_events,
        verbose=args.verbose,
    )

    if len(features) == 0:
        sys.exit(
            "[error] No events were processed. "
            "Check that your files contain events with ≥6 valid jets."
        )

    print(f"\n[info] Feature matrix: {features.shape}")
    print(f"[info] Events processed:  {stats['n_events_used']:,}")
    print(f"[info] Events skipped:    {stats['n_skipped_few']:,} (<6 jets), "
          f"{stats['n_skipped_nan']:,} (NaN/Inf)")
    print(f"[info] Splittings total:  {len(labels):,}")
    n_correct = int((labels == LABEL_CORRECT).sum())
    n_ambig   = int((labels == LABEL_AMBIG).sum())
    print(f"[info]   Correct:  {n_correct:,} ({100*n_correct/len(labels):.1f}%)")
    print(f"[info]   Ambiguous:{n_ambig:,} ({100*n_ambig/len(labels):.1f}%)")

    # t-SNE
    print("\n[info] Running t-SNE...")
    embedding = run_tsne(
        features,
        perplexity=args.perplexity,
        seed=args.seed,
        n_iter=args.n_iter,
        normalize=not args.no_normalize,
    )

    # Plot
    plot_tsne(embedding, labels, stats, args.output, args.perplexity)


if __name__ == "__main__":
    main()
