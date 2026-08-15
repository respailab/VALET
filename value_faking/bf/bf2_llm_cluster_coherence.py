"""BF2 upgrade - OpenAI embeddings + LLM-driven semantic clustering, matched against behavioral (gap-correlation) clustering."""
import json
import os
import sys

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from value_faking.paths import REPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from value_faking.bf.common import REPO
from bf2_coherence import value_definitions, mantel_test

if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None,
        epilog="Takes no options yet; paths are resolved by value_faking.paths.",
    ).parse_args()

EMBED_MODEL = "text-embedding-3-small"
GPT_CLUSTER_MODEL = "gpt-5.4-mini"
N_PERM = 10000
SEED = 0


def get_embeddings(defs: pd.Series) -> pd.DataFrame:
    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=defs.tolist())
    vecs = np.array([d.embedding for d in resp.data])
    sim = cosine_similarity(vecs)
    return pd.DataFrame(sim, index=defs.index, columns=defs.index)


CLUSTER_SYSTEM = """You are grouping personal/social value dimensions from a values survey into
semantically coherent higher-order clusters (similar in spirit to Schwartz's value
circumplex, e.g. openness-to-change vs conservation vs self-enhancement vs
self-transcendence - but decide the grouping and naming yourself from the
definitions given, don't just reproduce Schwartz's exact categories).

Return exactly this JSON and nothing else:
{
  "clusters": {
    "<short cluster name>": ["<value>", "<value>", ...],
    ...
  }
}
Every value given must appear in exactly one cluster. Aim for 4-7 clusters."""


def llm_cluster_values(defs: pd.Series, model: str = GPT_CLUSTER_MODEL) -> dict:
    client = OpenAI()
    listing = "\n\n".join(f"- {v}: {d[:400]}" for v, d in defs.items())
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLUSTER_SYSTEM},
            {"role": "user", "content": listing},
        ],
    )
    raw = resp.choices[0].message.content
    parsed = json.loads(raw)["clusters"]

    value_to_cluster = {}
    for cluster_name, values in parsed.items():
        for v in values:
            value_to_cluster[v] = cluster_name
    return value_to_cluster


def behavioral_clusters(gap_corr: pd.DataFrame, n_clusters: int) -> dict:
    values = gap_corr.index.tolist()
    dist = 1 - gap_corr.to_numpy()
    dist = (dist + dist.T) / 2  # symmetrize numerical noise
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return dict(zip(values, labels))


def match_and_score(semantic: dict, behavioral: dict, n_perm=N_PERM, seed=SEED):
    values = sorted(set(semantic) & set(behavioral))
    sem_labels, sem_names = pd.factorize([semantic[v] for v in values])
    beh_labels = np.array([behavioral[v] for v in values])

    ari = adjusted_rand_score(sem_labels, beh_labels)
    nmi = normalized_mutual_info_score(sem_labels, beh_labels)

    ct = pd.crosstab(pd.Series(sem_labels, name="semantic"), pd.Series(beh_labels, name="behavioral"))
    row_ind, col_ind = linear_sum_assignment(-ct.to_numpy())
    matched_overlap = ct.to_numpy()[row_ind, col_ind].sum()
    purity = matched_overlap / len(values)

    rng = np.random.default_rng(seed)
    null_ari = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(beh_labels)
        null_ari[i] = adjusted_rand_score(sem_labels, perm)
    p = (np.sum(null_ari >= ari) + 1) / (n_perm + 1)

    match_table = pd.DataFrame({
        "value": values,
        "semantic_cluster": [sem_names[l] for l in sem_labels],
        "behavioral_cluster": beh_labels,
    })

    return {
        "ari": ari, "nmi": nmi, "purity": purity, "perm_p": p,
        "contingency": ct, "match_table": match_table,
        "sem_names": sem_names,
    }


def main():
    defs = value_definitions()
    print(f"Loaded definitions for {len(defs)} values\n")

    print("=" * 78)
    print(f"1. Embeddings ({EMBED_MODEL}) -> semantic similarity matrix")
    print("=" * 78)
    sem_sim = get_embeddings(defs)
    sem_sim.to_csv(f"{REPO}/behavioral_analysis/bf2_semantic_similarity_embeddings.csv")
    print(f"  saved -> bf2_semantic_similarity_embeddings.csv")

    gap_corr = pd.read_csv(f"{REPO}/behavioral_analysis/bf2_gap_correlation_matrix.csv", index_col=0)
    obs_r, p, values = mantel_test(gap_corr, sem_sim)
    print(f"\n  Mantel test (embeddings-based semantic similarity): r={obs_r:+.3f}, p={p:.4f}")

    print("\n" + "=" * 78)
    print(f"2. LLM clustering ({GPT_CLUSTER_MODEL})")
    print("=" * 78)
    semantic = llm_cluster_values(defs)
    for cname in sorted(set(semantic.values())):
        members = [v for v, c in semantic.items() if c == cname]
        print(f"  {cname}: {members}")

    n_clusters = len(set(semantic.values()))
    print(f"\n  -> {n_clusters} semantic clusters")

    print("\n" + "=" * 78)
    print(f"3. Behavioral clustering (agglomerative on gap-correlation, k={n_clusters})")
    print("=" * 78)
    behavioral = behavioral_clusters(gap_corr, n_clusters)
    for k in sorted(set(behavioral.values())):
        members = [v for v, c in behavioral.items() if c == k]
        print(f"  cluster {k}: {members}")

    print("\n" + "=" * 78)
    print("4. Match + coherence")
    print("=" * 78)
    result = match_and_score(semantic, behavioral)
    print(f"  Adjusted Rand Index: {result['ari']:+.3f}")
    print(f"  Normalized Mutual Info: {result['nmi']:.3f}")
    print(f"  Purity (best-match overlap / n): {result['purity']:.3f}")
    print(f"  Permutation p (ARI, n_perm={N_PERM}): {result['perm_p']:.4f}\n")
    print(result["contingency"])

    result["match_table"].to_csv(f"{REPO}/behavioral_analysis/bf2_cluster_match.csv", index=False)
    with open(f"{REPO}/behavioral_analysis/bf2_semantic_clusters.json", "w") as f:
        json.dump(semantic, f, indent=2)
    print(f"\nSaved -> bf2_cluster_match.csv, bf2_semantic_clusters.json")


if __name__ == "__main__":
    main()
