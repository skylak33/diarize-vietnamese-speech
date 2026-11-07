"""Oracle diagnostics for complete AHC merge traces and hierarchy cuts."""

from collections import Counter
import math

from diarize.cluster import _cut_ahc_hierarchy
from experiments.analyze_partition_quality import partition_quality


def _composition(labels, members):
    return Counter(labels[index] for index in members if labels[index] is not None)


def _purity(composition):
    total = sum(composition.values())
    return None if not total else max(composition.values()) / total


def _entropy(composition):
    total = sum(composition.values())
    return None if not total else -sum(
        (count / total) * math.log2(count / total)
        for count in composition.values() if count)


def trace_reference_merges(children, distances, reference_labels, selected_n_clusters):
    """Trace oracle speaker composition and cut capacity of a complete AHC tree."""
    children = [list(pair) for pair in children]
    distances = list(distances)
    reference_labels = list(reference_labels)
    n_samples = len(reference_labels)
    if n_samples < 1 or len(children) != n_samples - 1 or len(distances) != len(children):
        raise ValueError("complete hierarchy requires n_samples - 1 children and distances")
    if not 1 <= selected_n_clusters <= n_samples:
        raise ValueError("selected_n_clusters must be between one and n_samples")
    if any(not math.isfinite(value) for value in distances) or any(
            right < left - 1e-12 for left, right in zip(distances, distances[1:])):
        raise ValueError("merge distances must be finite and nondecreasing")

    members = {index: [index] for index in range(n_samples)}
    performed = n_samples - selected_n_clusters
    merge_rows = []
    for step, ((left, right), distance) in enumerate(zip(children, distances)):
        if left not in members or right not in members or left == right:
            raise ValueError(f"invalid child identifiers at merge {step}")
        left_members, right_members = members[left], members[right]
        merged_members = left_members + right_members
        left_comp = _composition(reference_labels, left_members)
        right_comp = _composition(reference_labels, right_members)
        merged_comp = left_comp + right_comp
        reliable_before = sum(left_comp.values()) + sum(right_comp.values())
        purity_before = (None if not reliable_before else
                         (max(left_comp.values(), default=0)
                          + max(right_comp.values(), default=0)) / reliable_before)
        left_speakers, right_speakers = set(left_comp), set(right_comp)
        crossing = bool(left_speakers and right_speakers and
                        any(left_label != right_label
                            for left_label in left_speakers for right_label in right_speakers))
        merge_rows.append({
            "merge_index": step,
            "merge_distance": float(distance),
            "left_cluster_id": left,
            "right_cluster_id": right,
            "left_cluster_size": len(left_members),
            "right_cluster_size": len(right_members),
            "merged_cluster_size": len(merged_members),
            "reference_composition": dict(merged_comp),
            "reference_speaker_entropy_bits": _entropy(merged_comp),
            "left_purity": _purity(left_comp),
            "right_purity": _purity(right_comp),
            "purity_before_merge": purity_before,
            "purity_after_merge": _purity(merged_comp),
            "reference_crossing": crossing,
            "introduces_mixing": crossing and len(left_speakers) == len(right_speakers) == 1,
            "before_selected_cut": step < performed,
            "oracle_diagnostic": True,
        })
        members[n_samples + step] = merged_members

    cuts = []
    for n_clusters in range(1, n_samples + 1):
        predicted = _cut_ahc_hierarchy(children, n_samples, n_clusters).tolist()
        quality, _, _ = partition_quality(reference_labels, predicted)
        cuts.append({"n_clusters": n_clusters, "merges_performed": n_samples - n_clusters,
                     "ari": quality["ari"], "nmi": quality["nmi"],
                     "purity": quality["duration_weighted_cluster_purity"]})
    selected = next(row for row in cuts if row["n_clusters"] == selected_n_clusters)
    valid_ari = [row for row in cuts if row["ari"] is not None]
    oracle_best = max(valid_ari, key=lambda row: (row["ari"], row["purity"], -row["n_clusters"])) if valid_ari else None
    crossing_before = [row for row in merge_rows
                       if row["reference_crossing"] and row["before_selected_cut"]]
    boundary_tied = (0 < performed < len(distances)
                     and math.isclose(distances[performed - 1], distances[performed],
                                      rel_tol=1e-12, abs_tol=1e-12))
    summary = {
        "selected_n_clusters": selected_n_clusters,
        "selected_cut_position": performed,
        "selected_cut_metrics": selected,
        "selected_cut_tied_distance": boundary_tied,
        "earliest_reference_crossing_merge": next(
            (row["merge_index"] for row in merge_rows if row["reference_crossing"]), None),
        "cross_speaker_merges_before_selected_cut": len(crossing_before),
        "pure_clusters_before_selected_cut": sum(
            _purity(_composition(reference_labels, cluster_members)) == 1.0
            for cluster_id, cluster_members in _clusters_after(children, n_samples, performed).items()),
        "oracle_best_cut": oracle_best,
        "perfect_reference_partition_available": bool(
            oracle_best is not None and math.isclose(oracle_best["ari"], 1.0)),
        "failure_type": _failure_type(selected, oracle_best),
        "ambiguous_or_unlabeled_leaves": sum(label is None for label in reference_labels),
        "oracle_diagnostic": True,
    }
    return summary, merge_rows, cuts


def _clusters_after(children, n_samples, merges_performed):
    active = {index: [index] for index in range(n_samples)}
    for step, (left, right) in enumerate(children[:merges_performed]):
        active[n_samples + step] = active.pop(left) + active.pop(right)
    return active


def _failure_type(selected, oracle_best):
    if selected["ari"] is None or oracle_best is None:
        return "unresolved"
    if math.isclose(selected["ari"], 1.0):
        return "none"
    if math.isclose(oracle_best["ari"], 1.0):
        return "incorrect_hierarchy_cut"
    return "incorrect_early_merge"

