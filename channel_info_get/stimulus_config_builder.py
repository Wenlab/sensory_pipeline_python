"""Utilities for generating channel configuration and stimulus metadata."""

from __future__ import annotations

import colorsys
import itertools
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, cast

try:
    from utils.parse_stimulus_info import build_grouped_stimulus_entries, split_stimulus_name
except ImportError:  # pragma: no cover - allow script execution from sibling directory
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.parse_stimulus_info import build_grouped_stimulus_entries, split_stimulus_name

CONTROL_KEYWORDS = ("control",)
BUFFER_KEYWORDS = ("buffer",)
DEFAULT_SLICE_NUMBER = 20
DEFAULT_STATE_LENGTH = 8
GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


@dataclass(frozen=True)
class OdorEntry:
    identifier: str
    name: str


class OdorConfigError(RuntimeError):
    """Raised when the configuration cannot be generated."""


def generate_config_from_channel_meanings(
    channel_meanings: Mapping[str, str] | Mapping[int, str],
    odor_json_path: os.PathLike[str] | str,
    color_scheme_path: os.PathLike[str] | str,
    output_directory: os.PathLike[str] | str,
    *,
    bit_mode: str = "5-bit",
    slice_number: int = DEFAULT_SLICE_NUMBER,
    state_length: int = DEFAULT_STATE_LENGTH,
    buffer_prefixes: Optional[Iterable[Tuple[int, int, int]]] = None,
    stimulus_prefixes: Optional[Iterable[Tuple[int, int, int]]] = None,
    use_rgba: bool = False,  # New parameter: use RGBA colors
) -> Dict[str, Mapping[str, str] | int]:
    """Generate a channel configuration and update stimulus metadata.

    Parameters
    ----------
    channel_meanings:
        Mapping of channel identifiers to descriptive names. Keys may be strings
        or integers. Values that include the words "control" or "buffer"
        (case-insensitive) are treated as non-stimulus entries.
    odor_json_path:
        Path to the global stimulus metadata file. New stimuli are appended here
        with incrementing identifiers.
    color_scheme_path:
        Path to the color scheme JSON file. Newly created identifiers receive a
        distinct colour.
    output_directory:
        Destination directory for the generated ``config.json``.
    bit_mode:
        Either "5-bit" (default) or "16-bit". Determines how the valve bits in
        a state vector are interpreted.
    slice_number:
        Value written to the resulting configuration file.
    state_length:
        Number of bits in each state vector. Defaults to eight to match existing
        LabJack exports but can be overridden when necessary.
    buffer_prefixes / stimulus_prefixes:
        Optional explicit control-bit prefixes (first three bits of the state
        vector). When omitted, sensible defaults are used for the chosen
        ``bit_mode``.
    use_rgba:
        If True, generate RGBA colors with varying alpha for same-category stimuli.
        If False (default), generate HEX colors with varying brightness for same-category stimuli.

    Returns
    -------
    dict
        The configuration payload that was written to disk.

    Notes
    -----
    The function updates ``stimulus.json`` and ``stimulus_color_scheme.json`` in
    place. Identifiers follow the ``b{batch}_{index}`` format and continue from
    the highest existing entry.
    """

    normalized_channel_meanings = _normalize_channel_meanings(channel_meanings)
    stimuli = _extract_stimuli(normalized_channel_meanings)

    if not stimuli:
        raise OdorConfigError("No stimuli found in channel_meanings after excluding control and buffer entries.")

    odor_json_path = Path(odor_json_path)
    color_scheme_path = Path(color_scheme_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    odor_registry = _load_json_file(odor_json_path, default={})
    color_scheme = _load_json_file(color_scheme_path, default={})

    id_map = _assign_odor_ids(stimuli, odor_registry)
    
    # Use grouped color generation based on use_rgba parameter
    if use_rgba:
        _update_color_scheme_rgba(id_map.values(), color_scheme, odor_registry)
    else:
        _update_color_scheme_hex(id_map.values(), color_scheme, odor_registry)

    _write_json_file(odor_json_path, odor_registry)
    _write_json_file(color_scheme_path, color_scheme)

    config_payload = {
        "channel_meanings": normalized_channel_meanings,
        "state_mappings": _build_state_mappings(
            normalized_channel_meanings,
            id_map,
            bit_mode=bit_mode,
            state_length=state_length,
            buffer_prefixes=buffer_prefixes,
            stimulus_prefixes=stimulus_prefixes,
        ),
        "slice_number": slice_number,
    }

    config_path = output_directory / "config.json"
    _write_json_file(config_path, config_payload)

    return config_payload


def _group_identifiers_by_category(
    identifiers: Iterable[str],
    registry: Mapping[str, str]
) -> Dict[Tuple[int, str], List[Tuple[int, str]]]:
    """Group identifiers by batch and stimulus base name.
    
    Returns:
        Dict mapping (batch, base_name) to list of (index, identifier) tuples
    """
    groups: Dict[Tuple[int, str], List[Tuple[int, str]]] = defaultdict(list)
    
    for identifier in identifiers:
        try:
            batch, index = _parse_identifier(identifier)
            name = registry.get(identifier, "")
            base_name, _ = split_stimulus_name(name)
            group_key = (batch, base_name if base_name else name)
            groups[group_key].append((index, identifier))
        except OdorConfigError:
            continue
    
    # Sort each group by index
    for group_key in groups:
        groups[group_key].sort(key=lambda x: x[0])
    
    return groups


def _update_color_scheme_rgba(
    identifiers: Iterable[str],
    color_scheme: MutableMapping[str, str],
    registry: Mapping[str, str]
) -> None:
    """Update color scheme with RGBA colors - same base color but varying alpha for same category."""
    identifier_list = list(identifiers)
    groups = _group_identifiers_by_category(identifier_list, registry)
    
    # Generate base colors for each category
    category_base_colors: Dict[Tuple[int, str], Tuple[int, int, int]] = {}
    existing_colors = set(color_scheme.values())
    
    for idx, group_key in enumerate(sorted(groups.keys())):
        # Generate distinct base color for each category
        hue = (idx * GOLDEN_RATIO_CONJUGATE) % 1.0
        saturation = 0.7
        value = 0.85
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        base_rgb = tuple(int(c * 255) for c in rgb)
        category_base_colors[group_key] = base_rgb
    
    # Assign RGBA colors with varying alpha within each group
    for group_key, members in groups.items():
        base_rgb = category_base_colors[group_key]
        num_members = len(members)
        
        for member_idx, (_, identifier) in enumerate(members):
            if identifier in color_scheme:
                continue
            
            # Alpha varies from 1.0 (first) to 0.4 (last)
            if num_members == 1:
                alpha = 1.0
            else:
                alpha = 1.0 - (member_idx / (num_members - 1)) * 0.6
            
            rgba_str = f"rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {alpha:.2f})"
            color_scheme[identifier] = rgba_str
    
    _sort_mapping_by_identifier(color_scheme)


def _update_color_scheme_hex(
    identifiers: Iterable[str],
    color_scheme: MutableMapping[str, str],
    registry: Mapping[str, str]
) -> None:
    """Update color scheme with HEX colors - same base hue but varying brightness for same category."""
    identifier_list = list(identifiers)
    groups = _group_identifiers_by_category(identifier_list, registry)
    
    # Generate base hue for each category
    category_base_hues: Dict[Tuple[int, str], float] = {}
    
    for idx, group_key in enumerate(sorted(groups.keys())):
        hue = (idx * GOLDEN_RATIO_CONJUGATE) % 1.0
        category_base_hues[group_key] = hue
    
    # Assign HEX colors with varying brightness within each group
    for group_key, members in groups.items():
        base_hue = category_base_hues[group_key]
        num_members = len(members)
        
        for member_idx, (_, identifier) in enumerate(members):
            if identifier in color_scheme:
                continue
            
            # Saturation and value vary to create visual distinction
            if num_members == 1:
                saturation = 0.7
                value = 0.85
            else:
                # Value varies from 0.9 (brightest) to 0.5 (darkest)
                value = 0.9 - (member_idx / (num_members - 1)) * 0.4
                # Saturation varies slightly for better distinction
                saturation = 0.65 + (member_idx / max(num_members - 1, 1)) * 0.2
            
            rgb = colorsys.hsv_to_rgb(base_hue, saturation, value)
            hex_color = "#" + "".join(f"{int(channel * 255):02x}" for channel in rgb)
            color_scheme[identifier] = hex_color
    
    _sort_mapping_by_identifier(color_scheme)


def _normalize_channel_meanings(channel_meanings: Mapping[str, str] | Mapping[int, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in channel_meanings.items():
        normalized[str(key)] = value
    return normalized


def _extract_stimuli(channel_meanings: Mapping[str, str]) -> List[OdorEntry]:
    stimuli: List[OdorEntry] = []
    for key, value in channel_meanings.items():
        lower_value = value.lower()
        if any(keyword in lower_value for keyword in CONTROL_KEYWORDS + BUFFER_KEYWORDS):
            continue
        stimuli.append(OdorEntry(identifier=str(key), name=value))
    return stimuli


def _load_json_file(path: Path, *, default: MutableMapping[str, str]) -> MutableMapping[str, str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return default


def _write_json_file(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=True)


def _assign_odor_ids(
    stimuli: Iterable[OdorEntry],
    registry: MutableMapping[str, str],
) -> Dict[str, str]:
    existing_by_name = {name: identifier for identifier, name in registry.items()}
    group_registry, max_batch = _build_group_registry(registry)
    next_batch = max_batch + 1

    assigned: Dict[str, str] = {}

    stimuli_map = {entry.identifier: entry for entry in stimuli}
    grouped_entries = build_grouped_stimulus_entries(
        {entry.identifier: entry.name for entry in stimuli}
    )

    for group_data in grouped_entries:
        group_name = group_data["group"]
        group_entries = [
            stimuli_map[entry["code"]]
            for entry in group_data["entries"]
            if entry["code"] in stimuli_map
        ]
        if not group_entries:
            continue

        group_info = group_registry.get(group_name)
        if group_info is None:
            group_info = {"batch": next_batch, "indices": set()}
            group_registry[group_name] = group_info
            next_batch += 1

        batch = group_info["batch"]
        used_indices = cast(Set[int], group_info["indices"])

        for entry in group_entries:
            if entry.name in existing_by_name:
                assigned[entry.identifier] = existing_by_name[entry.name]
                continue

            index = _find_first_available_index(used_indices)
            identifier = _format_identifier(batch, index)
            registry[identifier] = entry.name
            assigned[entry.identifier] = identifier
            existing_by_name[entry.name] = identifier
            used_indices.add(index)

    _sort_mapping_by_identifier(registry)
    return assigned


def _parse_identifier(identifier: str) -> Tuple[int, int]:
    if not identifier.startswith("b") or "_" not in identifier:
        raise OdorConfigError(f"Unsupported stimulus identifier format: {identifier}")
    try:
        batch_part, index_part = identifier[1:].split("_", 1)
        return int(batch_part), int(index_part)
    except ValueError as exc:
        raise OdorConfigError(f"Invalid stimulus identifier: {identifier}") from exc


def _format_identifier(batch: int, index: int) -> str:
    return f"b{batch}_{index}"


def _identifier_sort_key(identifier: str) -> Tuple[int, int]:
    return _parse_identifier(identifier)


def _sort_mapping_by_identifier(mapping: MutableMapping[str, str]) -> None:
    ordered = dict(
        sorted(mapping.items(), key=lambda item: _identifier_sort_key(item[0]))
    )
    mapping.clear()
    mapping.update(ordered)


def _build_group_registry(
    registry: Mapping[str, str]
) -> Tuple[Dict[str, Dict[str, object]], int]:
    group_registry: Dict[str, Dict[str, object]] = {}
    max_batch = 0
    for identifier, name in registry.items():
        batch, index = _parse_identifier(identifier)
        max_batch = max(max_batch, batch)

        group_name, _ = split_stimulus_name(name)
        group_key = group_name or name

        info = group_registry.setdefault(
            group_key, {"batch": batch, "indices": set()}
        )

        if info["batch"] != batch:
            info["batch"] = min(info["batch"], batch)

        indices = cast(Set[int], info["indices"])
        indices.add(index)

    return group_registry, max_batch


def _find_first_available_index(used_indices: Set[int]) -> int:
    index = 0
    while index in used_indices:
        index += 1
    return index


def _generate_distinct_color(existing_colors: Iterable[str], *, seed: int) -> str:
    color_set = set(existing_colors)
    attempt = 0
    while True:
        hue = (seed + attempt * GOLDEN_RATIO_CONJUGATE) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.55, 0.92)
        candidate = "#" + "".join(f"{int(channel * 255):02x}" for channel in rgb)
        if candidate not in color_set:
            return candidate
        attempt += 1


def _build_state_mappings(
    channel_meanings: Mapping[str, str],
    id_map: Mapping[str, str],
    *,
    bit_mode: str,
    state_length: int,
    buffer_prefixes: Optional[Iterable[Tuple[int, int, int]]],
    stimulus_prefixes: Optional[Iterable[Tuple[int, int, int]]],
) -> Dict[str, str]:
    normalized_mode = bit_mode.strip().lower()
    if normalized_mode not in {"5-bit", "16-bit", "5bit", "16bit"}:
        raise OdorConfigError(f"Unsupported bit_mode: {bit_mode}")

    use_5_bit = "5" in normalized_mode
    stimulus_slots = _build_stimulus_slots(channel_meanings, use_5_bit)

    if use_5_bit:
        default_buffer_prefixes = [(1, 1, 1), (1, 0, 1), (0, 1, 1)]
        default_stimulus_prefixes = [(0, 1, 1)]
    else:
        default_buffer_prefixes = [(1, 1, 1), (1, 0, 1), (0, 1, 1)]
        default_stimulus_prefixes = [(0, 1, 1)]

    buffer_prefixes = list(buffer_prefixes) if buffer_prefixes is not None else default_buffer_prefixes
    stimulus_prefixes = list(stimulus_prefixes) if stimulus_prefixes is not None else default_stimulus_prefixes

    state_mappings: Dict[str, str] = {}
    zero_state = [0] * state_length
    state_mappings[_format_state(zero_state)] = "All Off"

    valve_bit_length = len(stimulus_slots[0][0])
    stimulus_pattern_map = {tuple(bits): channel_key for bits, channel_key in stimulus_slots}

    all_patterns = list(itertools.product([0, 1], repeat=valve_bit_length))

    for prefix in buffer_prefixes:
        for pattern in all_patterns:
            state = _compose_state(prefix, pattern, state_length)
            state_mappings[_format_state(state)] = "Buffer"

    for prefix in stimulus_prefixes:
        for pattern in all_patterns:
            state = _compose_state(prefix, pattern, state_length)
            if pattern in stimulus_pattern_map:
                channel_key = stimulus_pattern_map[pattern]
                identifier = id_map[channel_key]
                state_mappings[_format_state(state)] = identifier
            else:
                state_mappings[_format_state(state)] = "Buffer"

    return state_mappings


def _build_stimulus_slots(
    channel_meanings: Mapping[str, str],
    use_5_bit: bool,
) -> List[Tuple[List[int], str]]:
    if use_5_bit:
        stimulus_channels: List[Tuple[int, str]] = []
        for key, value in channel_meanings.items():
            if any(keyword in value.lower() for keyword in CONTROL_KEYWORDS + BUFFER_KEYWORDS):
                continue
            try:
                channel_index = int(key)
            except ValueError as exc:
                raise OdorConfigError(
                    "5-bit mode requires integer channel identifiers for stimulus valves."
                ) from exc
            stimulus_channels.append((channel_index, key))

        if not stimulus_channels:
            raise OdorConfigError("No stimulus channels available for 5-bit mode.")

        min_channel = min(index for index, _ in stimulus_channels)
        valve_count = max(index for index, _ in stimulus_channels) - min_channel + 1
        slots: List[Tuple[List[int], str]] = []
        for index, key in sorted(stimulus_channels):
            valve_bits = [0] * valve_count
            position = index - min_channel
            valve_bits[position] = 1
            slots.append((valve_bits, key))
        return slots

    slots_16: List[Tuple[List[int], str]] = []
    for key, value in channel_meanings.items():
        if any(keyword in value.lower() for keyword in CONTROL_KEYWORDS + BUFFER_KEYWORDS):
            continue
        valve_bits = _decode_16bit_key(key)
        slots_16.append((valve_bits, key))

    slots_16.sort(key=lambda item: _decode_16bit_numeric(item[1]))
    return slots_16


def _compose_state(prefix: Tuple[int, int, int], valve_bits: Iterable[int], state_length: int) -> List[int]:
    state = list(prefix)
    state.extend(valve_bits)
    residual = state_length - len(state)
    if residual < 0:
        raise OdorConfigError("State length shorter than composed bit pattern.")
    state.extend([0] * residual)
    return state


def _format_state(state: Iterable[int]) -> str:
    return "[" + ", ".join(str(bit) for bit in state) + "]"


def _decode_16bit_key(key: str) -> List[int]:
    numeric = _decode_16bit_numeric(key)
    if numeric < 0 or numeric >= 16:
        raise OdorConfigError(
            f"16-bit mode expects identifiers in range 0-15 (got {numeric} from {key!r})."
        )
    return [(numeric >> bit) & 1 for bit in range(4)]


def _decode_16bit_numeric(key: str) -> int:
    try:
        return int(key, 10)
    except ValueError:
        try:
            return int(key, 16)
        except ValueError as exc:
            raise OdorConfigError(
                "16-bit mode requires decimal or hexadecimal keys for stimulus entries."
            ) from exc