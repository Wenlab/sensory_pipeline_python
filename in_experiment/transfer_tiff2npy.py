import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import re

from utils.read_vols_using_dask import (
    save_volumes_with_tiff_range,
    extract_volume_numbers_from_dir
)


def _resolve_worm_output_path(base_folder, worm_lower):
    """Return the resolved output path for a worm and ensure folder structure."""
    base_path = Path(base_folder)
    parent = base_path.parent

    if parent.name.lower() == worm_lower.lower():
        worm_root = parent
        resolved = base_path
    else:
        worm_root = parent / worm_lower
        resolved = worm_root / base_path.name

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved, worm_root


def _get_camera_subfolder(camera_type):
    """Get camera subfolder name based on camera type."""
    camera_folders = {
        "red": "0_Camera-Red_VSC-10629",
        "green": "1_Camera-Green_VSC-09321"
    }
    if camera_type not in camera_folders:
        raise ValueError(f"camera_type must be 'red' or 'green', got '{camera_type}'")
    return camera_folders[camera_type]


def _extract_date_info(folder_name):
    """
    Extract date information from folder name.
    Expected format: w1_2025-10-22_10-21-21
    Returns: YYYYMM format string
    """
    try:
        parts = folder_name.split("_")
        if len(parts) >= 2:
            date_str = parts[1]  # e.g., "2025-10-22"
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%Y%m")
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse date from '{folder_name}': {e}")
    return "202510"  # fallback


def _get_volume_read_params():
    """Get standard volume reading parameters."""
    return dict(
        z_start_frame_number=0,
        z_end_frame_number=17,
        mod2_reverse=[False, False],
        img_width=1024,
        img_height=1024,
        frame_number_per_volume=20,
        img_dtype=np.uint16,
    )


def _group_folders_by_worm(ex_folder_path):
    """
    Group folders by worm name and type (ref vs ex).
    
    Parameters:
    -----------
    ex_folder_path : Path
        Path to NAS directory
        
    Returns:
    --------
    dict : {worm_name_lower: {'ref': [folders], 'ex': [folders]}}
    """
    # Find all folders starting with w or W followed by digits
    all_folders = [
        f for f in ex_folder_path.iterdir() 
        if f.is_dir() and re.match(r'^[wW]\d+_', f.name)
    ]
    
    if not all_folders:
        print(f"Warning: No folders matching pattern '[wW]<digits>_*' found in {ex_folder_path}")
        return {}
    
    # Group by worm name (case-insensitive)
    groups = {}
    for folder in all_folders:
        # Extract worm identifier (e.g., "w1" or "W1")
        worm_token = folder.name.split("_")[0]
        worm_lower = worm_token.lower()
        
        # Uppercase = ref, lowercase = ex
        folder_type = "ref" if worm_token[0].isupper() else "ex"
        
        if worm_lower not in groups:
            groups[worm_lower] = {"ref": [], "ex": []}
        groups[worm_lower][folder_type].append(folder)
    
    # Sort folders by name (timestamp makes this chronological)
    for worm_lower in groups:
        groups[worm_lower]["ref"].sort(key=lambda p: p.name)
        groups[worm_lower]["ex"].sort(key=lambda p: p.name)
    
    return groups


def _process_ref_volumes(
    ref_folders,
    ref_output_folder,
    worm_lower,
    date_info,
    camera_type,
    volume_read_params,
    show_progress
):
    """
    Process reference volumes - one volume per folder.
    
    Each ref folder contributes exactly one volume (the first available),
    saved as: ImgStk001_dk001_{worm}_Dt{YYYYMM}_{seq:06d}.npy
    """
    if not ref_folders:
        return
    
    print(f"\nProcessing {len(ref_folders)} reference volume(s) for {worm_lower}...")
    
    camera_subfolder = _get_camera_subfolder(camera_type)
    frame_number_per_volume = volume_read_params["frame_number_per_volume"]
    z_start = volume_read_params["z_start_frame_number"]
    z_end = volume_read_params["z_end_frame_number"]
    
    output_path, _ = _resolve_worm_output_path(ref_output_folder, worm_lower)
    if show_progress:
        tqdm.write(f"  Output (ref) -> {output_path}")
    
    iterator = tqdm(ref_folders, total=len(ref_folders), desc="Ref volumes") if show_progress else ref_folders
    sequence_counter = 0
    
    for ref_folder in iterator:
        tiff_dir = ref_folder / camera_subfolder
        
        if not tiff_dir.exists():
            print(f"  Warning: Camera folder not found: {tiff_dir}")
            continue
        
        # Find first available volume
        available_vols = extract_volume_numbers_from_dir(str(tiff_dir))
        if not available_vols:
            print(f"  Warning: No volumes found in {tiff_dir}")
            continue
        
        first_vol = min(available_vols)
        
        # Calculate tiff range for this volume
        start_tiff = first_vol * frame_number_per_volume + z_start
        end_tiff = first_vol * frame_number_per_volume + z_end
        
        saved_files = save_volumes_with_tiff_range(
            tiff_path_=str(tiff_dir),
            volume_read_params=volume_read_params,
            output_folder=str(output_path),
            worm_name=worm_lower,
            date_info=date_info,
            start_tiff_number=start_tiff,
            end_tiff_number=end_tiff,
            show_progress=False,
            filename_pattern="ImgStk001_dk001_{worm}_Dt{date}_{seq:06d}.npy",
            sequence_start=sequence_counter,
        )
        saved_count = len(saved_files)
        sequence_counter += saved_count

        if saved_count == 0:
            print(f"  Warning: No volumes saved for reference folder {ref_folder.name}")
            continue

        if show_progress:
            tqdm.write(
                f"  ✓ Saved {saved_count} reference volume(s) from {ref_folder.name}; "
                f"latest file: {saved_files[-1].name}"
            )

    print(f"✓ Completed {sequence_counter} reference volume(s)")


def _process_ex_volumes(
    ex_folders,
    ex_output_folder,
    worm_lower,
    date_info,
    camera_type,
    volume_read_params,
    show_progress
):
    """
    Process experiment volumes - all volumes from each folder.
    
    Each ex folder gets its own subfolder:
      ImgStk001_dk{idx:03d}_{worm}_Dt{YYYYMM}/
        containing files:
            ImgStk001_dk{idx:03d}_{worm}_Dt{YYYYMM}_{seq:06d}.npy
    """
    if not ex_folders:
        return
    
    print(f"\nProcessing {len(ex_folders)} experiment folder(s) for {worm_lower}...")
    
    camera_subfolder = _get_camera_subfolder(camera_type)
    
    ex_output_base, _ = _resolve_worm_output_path(ex_output_folder, worm_lower)
    if show_progress:
        tqdm.write(f"  Output (ex) -> {ex_output_base}")
    
    iterator = tqdm(enumerate(ex_folders, start=1), total=len(ex_folders), desc="Ex folders") if show_progress else enumerate(ex_folders, start=1)
    total_saved = 0
    
    for idx, ex_folder in iterator:
        tiff_dir = ex_folder / camera_subfolder
        
        if not tiff_dir.exists():
            print(f"  Warning: Camera folder not found: {tiff_dir}")
            continue
        
        # Create subfolder for this experiment under the worm-specific folder
        subfolder_name = f"ImgStk001_dk{idx:03d}_{worm_lower}_Dt{date_info}"
        subfolder_path = ex_output_base / subfolder_name
        subfolder_path.mkdir(parents=True, exist_ok=True)
        
        if show_progress:
            tqdm.write(f"\n  Processing: {ex_folder.name} -> {subfolder_name}")
        
        # Save all volumes using sequential naming within the subfolder
        saved_files = save_volumes_with_tiff_range(
            tiff_path_=str(tiff_dir),
            volume_read_params=volume_read_params,
            output_folder=str(subfolder_path),
            worm_name=worm_lower,
            date_info=date_info,
            start_tiff_number=0,
            end_tiff_number=None,
            show_progress=False,
            filename_pattern="{subfolder_name}_{seq:06d}.npy",
            sequence_start=0,
            extra_format_kwargs={"subfolder_name": subfolder_name, "folder_idx": idx},
        )
        saved_count = len(saved_files)
        total_saved += saved_count

        if saved_count == 0:
            print(f"  Warning: No experiment volumes saved for folder {ex_folder.name}")
            continue

        if show_progress:
            tqdm.write(
                f"  ✓ Saved {saved_count} volume(s) to {subfolder_name} (under {ex_output_base})"
            )
    
    print(f"✓ Completed {len(ex_folders)} experiment folder(s) with {total_saved} volume(s)")


def transfer_tiff2npy(
    ex_folder_path,
    ref_output_folder,
    ex_output_folder,
    camera_type="red",
    show_progress=True
):
    """
    Transfer TIFF files from ex_folder_path to NPY format, separating reference and experiment volumes.
    
    This function processes folders with naming convention: {WORM_ID}_{YYYY-MM-DD}_{HH-MM-SS}
    - Uppercase worm ID (e.g., "W1") = reference volume folders
    - Lowercase worm ID (e.g., "w1") = experiment volume folders
    
    Reference volumes:
    ------------------
    - One volume (first available) is extracted from each ref folder
    - Saved as: ImgStk001_dk001_{worm}_Dt{YYYYMM}_{seq:06d}.npy (sequential per worm)
    - Stored under: <parent_of_ref_output>/<worm>/<name_of_ref_output_folder>
    
    Experiment volumes:
    -------------------
    - All volumes are extracted from each ex folder
    - Each folder gets its own subfolder: ImgStk001_dk{sort_idx:03d}_{worm}_Dt{YYYYMM}/
    - Files named: ImgStk001_dk{sort_idx:03d}_{worm}_Dt{YYYYMM}_{seq:06d}.npy (sequence resets per folder)
    - Stored under: <parent_of_ex_output>/<worm>/<name_of_ex_output_folder>/ImgStk001_dkXXX_...
    
    Parameters:
    -----------
    ex_folder_path : str or Path
        Path to ex_folder_path folder containing experiment folders
    ref_output_folder : str or Path
        Output directory for reference volumes
    ex_output_folder : str or Path
        Output directory for experiment volumes
    camera_type : str, default="red"
        Camera type to process: "red" or "green"
    show_progress : bool, default=True
        Whether to display progress bars
        
    Example:
    --------
    >>> transfer_tiff2npy(
    ...     nas_folder_path=r"\\NAS\experiments",
    ...     ref_output_folder=r"H:\output\ref_volumes",
    ...     ex_output_folder=r"H:\output\ex_volumes",
    ...     camera_type="red"
    ... )
    """
    nas_path = Path(ex_folder_path)
    
    if not nas_path.exists():
        raise FileNotFoundError(f"NAS folder not found: {ex_folder_path}")
    
    print(f"\n{'='*70}")
    print(f"Transfer TIFF to NPY")
    print(f"{'='*70}")
    print(f"Source: {ex_folder_path}")
    print(f"Camera: {camera_type}")
    print(f"Ref output: {ref_output_folder}")
    print(f"Ex output: {ex_output_folder}")
    print(f"{'='*70}\n")
    
    # Group folders by worm
    worm_groups = _group_folders_by_worm(nas_path)
    
    if not worm_groups:
        print("No valid folders found. Exiting.")
        return
    
    print(f"Found {len(worm_groups)} worm(s) to process:")
    for worm_name, folders in worm_groups.items():
        print(f"  - {worm_name}: {len(folders['ref'])} ref + {len(folders['ex'])} ex folders")
    
    # Get volume reading parameters
    volume_read_params = _get_volume_read_params()
    
    # Process each worm
    for worm_lower, folders_dict in worm_groups.items():
        print(f"\n{'─'*70}")
        print(f"Processing worm: {worm_lower.upper()}")
        print(f"{'─'*70}")
        
        # Extract date info from first available folder
        sample_folders = folders_dict['ref'] + folders_dict['ex']
        date_info = _extract_date_info(sample_folders[0].name) if sample_folders else "202510"
        
        # Process reference volumes
        if folders_dict['ref']:
            _process_ref_volumes(
                ref_folders=folders_dict['ref'],
                ref_output_folder=ref_output_folder,
                worm_lower=worm_lower,
                date_info=date_info,
                camera_type=camera_type,
                volume_read_params=volume_read_params,
                show_progress=show_progress
            )
        
        # Process experiment volumes
        if folders_dict['ex']:
            _process_ex_volumes(
                ex_folders=folders_dict['ex'],
                ex_output_folder=ex_output_folder,
                worm_lower=worm_lower,
                date_info=date_info,
                camera_type=camera_type,
                volume_read_params=volume_read_params,
                show_progress=show_progress
            )
    
    print(f"\n{'='*70}")
    print(f"✓ Transfer completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    nas_folder_path = r"\\192.168.1.192\Odor\Jinghao-Wang\20251025_ZM11706_salt"
    ref_output_folder = r"I:\WJH\infer\me\20251025\ZM\ref_volumes"
    ex_output_folder = r"I:\WJH\infer\me\20251025\ZM\ex_volumes"
    
    transfer_tiff2npy(
        ex_folder_path=nas_folder_path,
        ref_output_folder=ref_output_folder,
        ex_output_folder=ex_output_folder,
        camera_type="green",
        show_progress=True
    )
