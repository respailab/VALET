"""BF2 - Is the faking coherent or noise?"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from value_faking.bf.common import REPO, available_main_models, value_gap_table
from value_faking.paths import REPO

if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None,
        epilog="Takes no options yet; paths are resolved by value_faking.paths.",
    ).parse_args()

N_PERM = 10000
SEED = 0


def value_definitions() -> pd.Series:
    frames = []
    for model, path in available_main_models().items():
        df = pd.read_csv(path)
        frames.append(df[["value", "topic"]])
    all_topics = pd.concat(frames, ignore_index=True)
    return all_topics.groupby("value")["topic"].apply(
        lambda s: " ".join(sorted(set(s.dropna().astype(str))))
    )


def semantic_similarity_matrix(defs: pd.Series) -> pd.DataFrame:
    vec = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vec.fit_transform(defs.values)
    sim = cosine_similarity(X)
    return pd.DataFrame(sim, index=defs.index, columns=defs.index)


def gap_correlation_matrix(gap_table: pd.DataFrame) -> pd.DataFrame:
    return gap_table.corr(method="spearman")


def upper_tri(mat: pd.DataFrame, values: list) -> np.ndarray:
    m = mat.loc[values, values].to_numpy()
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


def mantel_test(mat_a: pd.DataFrame, mat_b: pd.DataFrame, n_perm=N_PERM, seed=SEED):
    values = sorted(set(mat_a.index) & set(mat_b.index))
    a = upper_tri(mat_a, values)
    b = upper_tri(mat_b, values)
    obs_r = stats.pearsonr(a, b).statistic

    rng = np.random.default_rng(seed)
    n = len(values)
    null = np.empty(n_perm)
    b_mat = mat_b.loc[values, values].to_numpy()
    for i in range(n_perm):
        perm = rng.permutation(n)
        b_perm = b_mat[np.ix_(perm, perm)]
        iu = np.triu_indices_from(b_perm, k=1)
        null[i] = stats.pearsonr(a, b_perm[iu]).statistic

    p = (np.sum(np.abs(null) >= np.abs(obs_r)) + 1) / (n_perm + 1)
    return obs_r, p, values


def main():
    gap_table = value_gap_table()  # model x value
    print(f"Gap table: {gap_table.shape[0]} models x {gap_table.shape[1]} values\n")

    gap_corr = gap_correlation_matrix(gap_table)

    defs = value_definitions()
    sem_sim = semantic_similarity_matrix(defs)

    obs_r, p, values = mantel_test(gap_corr, sem_sim)

    print("=" * 78)
    print("MANTEL TEST: gap-correlation matrix vs semantic-similarity matrix")
    print("=" * 78)
    print(f"  n values compared: {len(values)}")
    print(f"  observed r: {obs_r:+.3f}")
    print(f"  permutation p (label shuffle, n_perm={N_PERM}): {p:.4f}\n")

    print("=" * 78)
    print("TOP 15 most gap-correlated value pairs (candidates for 'fake together')")
    print("=" * 78)
    pairs = []
    for i, v1 in enumerate(values):
        for v2 in values[i + 1:]:
            pairs.append((v1, v2, gap_corr.loc[v1, v2], sem_sim.loc[v1, v2]))
    pairs_df = pd.DataFrame(pairs, columns=["value_1", "value_2", "gap_corr", "semantic_sim"])
    print(pairs_df.sort_values("gap_corr", ascending=False).head(15).to_string(index=False))

    print()
    print("=" * 78)
    print("Correlation between gap_corr and semantic_sim across all value pairs (sanity check)")
    print("=" * 78)
    rho = stats.spearmanr(pairs_df["gap_corr"], pairs_df["semantic_sim"]).statistic
    print(f"  spearman rho (pairwise, not Mantel-corrected): {rho:+.3f}")

    gap_corr.to_csv(f"{REPO}/behavioral_analysis/bf2_gap_correlation_matrix.csv")
    sem_sim.to_csv(f"{REPO}/behavioral_analysis/bf2_semantic_similarity_matrix.csv")
    pairs_df.to_csv(f"{REPO}/behavioral_analysis/bf2_pairs.csv", index=False)
    print(f"\nSaved matrices + pairs -> behavioral_analysis/bf2_*.csv")


if __name__ == "__main__":
    main()
