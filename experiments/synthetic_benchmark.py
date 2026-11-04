"""Deterministic, leakage-safe splits for synthetic benchmark manifests.

Input records require ``source_file``; callers receiving ``source_locator`` from
an audit must normalize it to that field before using this module.
"""

import hashlib
import json
import math
from copy import deepcopy


IDENTITY_FIELDS = (
    "speaker_id",
    "source_file",
    "content_sha256",
    "source_recording_id",
)
SPLITS = ("development", "validation", "holdout")
DEFAULT_SPLIT_RATIOS = {"development": 0.6, "validation": 0.2, "holdout": 0.2}
ASSIGNMENT_ALGORITHM = "sha256-connected-identity-components-v1"
CHECK_NAMES = {
    "speaker_id": "speaker_disjoint",
    "source_file": "source_file_disjoint",
    "content_sha256": "content_hash_disjoint",
    "source_recording_id": "source_recording_disjoint",
}


def _validate_records(records, require_split=False):
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        fields = IDENTITY_FIELDS + (("split",) if require_split else ())
        missing = [field for field in fields if field not in record]
        if missing:
            raise ValueError(f"records[{index}] is missing fields: {missing}")
        invalid = [field for field in fields if not isinstance(record[field], str) or not record[field].strip()]
        if invalid:
            raise ValueError(f"records[{index}] has empty/non-string fields: {invalid}")
        digest = record["content_sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError(f"records[{index}].content_sha256 must be a hexadecimal SHA-256 digest")


def _ratios(split_ratios):
    ratios = DEFAULT_SPLIT_RATIOS if split_ratios is None else split_ratios
    if not isinstance(ratios, dict) or not ratios:
        raise ValueError("split_ratios must be a non-empty object")
    if set(ratios) != set(SPLITS):
        raise ValueError(f"split_ratios must contain exactly {list(SPLITS)}")
    if any(not isinstance(name, str) or not name.strip() for name in ratios):
        raise ValueError("split names must be non-empty strings")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in ratios.values()):
        raise ValueError("split ratios must be positive numbers")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    return {name: float(ratios[name]) for name in SPLITS}


def assign_splits(records, seed=42, split_ratios=None):
    """Assign connected speaker/source identities to deterministic splits."""
    _validate_records(records)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    ratios = _ratios(split_ratios)

    parents = list(range(len(records)))

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    seen = {}
    for index, record in enumerate(records):
        for field in IDENTITY_FIELDS:
            value = record[field].lower() if field == "content_sha256" else record[field]
            identity = (field, value)
            if identity in seen:
                union(index, seen[identity])
            else:
                seen[identity] = index

    components = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    boundaries = []
    cumulative = 0.0
    for name, ratio in ratios.items():
        cumulative += ratio
        boundaries.append((cumulative, name))

    assigned = [dict(record) for record in records]
    for indexes in components.values():
        identities = sorted({
            f"{field}={records[index][field].lower() if field == 'content_sha256' else records[index][field]}"
            for index in indexes for field in IDENTITY_FIELDS
        })
        payload = json.dumps([seed, identities], ensure_ascii=False, separators=(",", ":"))
        score = int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16) / 2**256
        split = next(name for boundary, name in boundaries if score < boundary)
        for index in indexes:
            assigned[index]["split"] = split

    validate_split_leakage(assigned)
    return assigned


def validate_split_leakage(records):
    """Fail if any speaker or source identity occurs in more than one split."""
    _validate_records(records, require_split=True)
    checks = {}
    for field in IDENTITY_FIELDS:
        split_by_value = {}
        for record in records:
            value = record[field].lower() if field == "content_sha256" else record[field]
            previous = split_by_value.setdefault(value, record["split"])
            if previous != record["split"]:
                raise ValueError(f"Cross-split {field} leakage: {record[field]!r}")
        checks[CHECK_NAMES[field]] = True
    return checks


def create_split_manifest(records, provenance, seed=42, split_ratios=None):
    """Return a deterministic JSON-compatible manifest with source provenance."""
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("provenance must be a non-empty object")
    ratios = _ratios(split_ratios)
    assigned = assign_splits(records, seed=seed, split_ratios=ratios)
    checks = validate_split_leakage(assigned)
    by_split = {split: sum(record["split"] == split for record in assigned) for split in ratios}
    empty_splits = [split for split, count in by_split.items() if not count]
    if empty_splits:
        raise ValueError(f"Cannot create benchmark manifest with empty splits: {empty_splits}")
    manifest = {
        "schema_version": 1,
        "provenance": deepcopy(provenance),
        "assignment": {
            "algorithm": ASSIGNMENT_ALGORITHM,
            "seed": seed,
            "split_ratios": ratios,
        },
        "checks": checks,
        "summary": {
            "record_count": len(assigned),
            "speaker_count": len({record["speaker_id"] for record in assigned}),
            "by_split": by_split,
        },
        "records": sorted(assigned, key=lambda record: json.dumps(record, ensure_ascii=False, sort_keys=True)),
    }
    json.dumps(manifest, sort_keys=True)
    return manifest
