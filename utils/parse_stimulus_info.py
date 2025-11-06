"""Helpers for grouping and sorting stimulus descriptors."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple


def parse_concentration(concentration_str: str, *_: object) -> Optional[Tuple[int, float]]:
    """Return an ordering key for recognised concentration strings.

    Additional positional arguments are ignored for backwards compatibility.
    """

    text = concentration_str.strip()
    if not text:
        return None

    match = re.search(r"([\d.]+)\s*%", text)
    if match:
        return 0, float(match.group(1))

    match = re.search(r"([\d.]+)\s*uM", text, re.IGNORECASE)
    if match:
        return 1, float(match.group(1))

    match = re.search(r"([\d.]+)\s*mM", text, re.IGNORECASE)
    if match:
        return 2, float(match.group(1))

    match = re.search(r"E(-?\d+)", text, re.IGNORECASE)
    if match:
        return 3, -float(match.group(1))

    return None


def split_stimulus_name(full_name: str) -> Tuple[str, str]:
    """Split a stimulus label into a base group and a variant suffix."""

    cleaned = full_name.strip()
    if not cleaned:
        return "", ""

    parts = cleaned.split()
    if len(parts) == 1:
        return cleaned, ""

    for start in range(len(parts) - 1, -1, -1):
        candidate_suffix = " ".join(parts[start:])
        if parse_concentration(candidate_suffix) is not None:
            group_part = " ".join(parts[:start]).strip()
            if not group_part:
                return cleaned, candidate_suffix
            return group_part, candidate_suffix

    group_part = " ".join(parts[:-1]).strip()
    if not group_part:
        return cleaned, ""
    return group_part, parts[-1]


def build_grouped_stimulus_entries(
    stimulus_info_dict: Mapping[str, str]
) -> List[Dict[str, object]]:
    """Return grouped stimulus metadata with deterministic ordering."""

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for code, full_name in stimulus_info_dict.items():
        group_name, variant = split_stimulus_name(full_name)
        group_key = group_name or full_name
        entry = {
            "code": str(code),
            "name": full_name,
            "variant": variant,
            "concentration": parse_concentration(full_name),
        }
        grouped.setdefault(group_key, []).append(entry)

    grouped_entries: List[Dict[str, object]] = []
    for group_key, entries in grouped.items():
        entries.sort(key=_stimulus_entry_sort_key)
        grouped_entries.append({"group": group_key, "entries": entries})

    grouped_entries.sort(key=lambda item: item["group"])
    return grouped_entries


def group_and_sort_stimuli(stimulus_info_dict: Mapping[str, str]) -> List[Tuple[str, List[str]]]:
    """Group stimulus codes by compound and sort each group consistently."""

    grouped_entries = build_grouped_stimulus_entries(stimulus_info_dict)
    return [
        (group_data["group"], [entry["code"] for entry in group_data["entries"]])
        for group_data in grouped_entries
    ]


def _stimulus_entry_sort_key(entry: Mapping[str, object]) -> Tuple:
    concentration = entry.get("concentration")
    if concentration is not None:
        priority, value = concentration
        return 0, priority, value

    code = str(entry.get("code", ""))
    code_key = _code_sort_key(code)
    variant = str(entry.get("variant", ""))
    return (1,) + code_key + (variant,)


def _code_sort_key(code: str) -> Tuple[int, object]:
    try:
        return 0, int(code)
    except (TypeError, ValueError):
        pass

    try:
        return 1, int(code, 16)
    except (TypeError, ValueError):
        pass

    numeric_parts = re.findall(r"\d+", code)
    if numeric_parts:
        return 2, int(numeric_parts[0])

    return 3, code
