"""
Generate video from TIFF files with stimulus state labels
"""
import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import tifffile
import pandas as pd
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_uint16_to_uint8(img, contrast_limits=(100, 255)):
    """
    Normalize uint16 image to uint8 with contrast limits
    
    Parameters:
    -----------
    img : np.ndarray
        Input image (uint16)
    contrast_limits : tuple
        (min, max) values for contrast adjustment
    
    Returns:
    --------
    np.ndarray
        Normalized image (uint8)
    """
    img = img.astype(np.float32)
    vmin, vmax = contrast_limits
    
    # Clip values
    img = np.clip(img, vmin, vmax)
    
    # exclude extreme values
    img[img < vmin] = vmin
    img[img > vmax] = vmax

    # Normalize to 0-255
    img = ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    return img


def get_state_at_frame(frame_idx, stimulus_df, stimulus_mapping=None):
    """
    Get the stimulus state for a given frame index
    
    Parameters:
    -----------
    frame_idx : int
        Current frame index (0-based)
    stimulus_df : pd.DataFrame
        DataFrame with columns 'start', 'end', 'state'
    stimulus_mapping : dict, optional
        Dictionary mapping state codes to names
    
    Returns:
    --------
    str
        Current state name
    """
    for _, row in stimulus_df.iterrows():
        if row['start'] <= frame_idx < row['end']:
            state = str(row['state'])
            if stimulus_mapping and state in stimulus_mapping:
                return stimulus_mapping[state]
            return state
    return 'Unknown'


def add_text_to_frame(frame, text, position='top_right', 
                      font=cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale=1.0, 
                      color=(255, 255, 255), 
                      thickness=2,
                      bg_color=(0, 0, 0),
                      padding=10):
    """
    Add text with background to frame
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame
    text : str
        Text to add
    position : str
        Position of text ('top_right', 'top_left', 'bottom_right', 'bottom_left')
    font : int
        OpenCV font type
    font_scale : float
        Font scale
    color : tuple
        Text color (B, G, R) for color images or intensity for grayscale
    thickness : int
        Text thickness
    bg_color : tuple
        Background color
    padding : int
        Padding around text
    
    Returns:
    --------
    np.ndarray
        Frame with text
    """
    # Make a copy to avoid modifying original
    frame = frame.copy()
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Calculate position
    h, w = frame.shape[:2]
    
    if position == 'top_right':
        x = w - text_width - padding * 2
        y = padding + text_height
    elif position == 'top_left':
        x = padding
        y = padding + text_height
    elif position == 'bottom_right':
        x = w - text_width - padding * 2
        y = h - padding
    elif position == 'bottom_left':
        x = padding
        y = h - padding
    else:
        x = padding
        y = padding + text_height
    
    # Draw background rectangle
    cv2.rectangle(
        frame, 
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color, 
        -1
    )
    
    # Draw text
    cv2.putText(
        frame, 
        text, 
        (x, y), 
        font, 
        font_scale, 
        color, 
        thickness
    )
    
    return frame


def tiff_to_video(
    parent_folder,
    output_video_path,
    stimulus_df,
    worm_key=None,
    contrast_limits=(100, 255),
    fps=2.5,
    codec='mp4v',
    text_position='top_right',
    font_scale=1.0,
    text_color=(255, 255, 255),
    text_bg_color=(0, 0, 0),
    frame_pattern='*.tif',
    start_frame=None,
    end_frame=None,
    roi=None,
    stimulus_mapping_file=None,
):
    """
    Generate video from TIFF files with stimulus state labels
    
    Parameters:
    -----------
    parent_folder : str
        Path to folder containing TIFF files or subfolders with TIFF files
    output_video_path : str
        Path to output video file
    stimulus_df : pd.DataFrame or dict
        If DataFrame: must contain 'start', 'end', 'state' columns
        If dict: dictionary where keys are worm IDs and values are DataFrames
    worm_key : str, optional
        If stimulus_df is a dict, specify which worm's data to use
    contrast_limits : tuple
        (min, max) values for contrast adjustment from uint16
    fps : int
        Frames per second for output video
    codec : str
        Video codec ('mp4v', 'XVID', 'H264', etc.)
    text_position : str
        Position of state label ('top_right', 'top_left', etc.)
    font_scale : float
        Font scale for text
    text_color : tuple
        Text color (B, G, R) for color or intensity for grayscale
    text_bg_color : tuple
        Background color for text
    frame_pattern : str
        Pattern to match TIFF files (e.g., '*.tif', '*.tiff')
    start_frame : int, optional
        Starting frame index (0-based)
    end_frame : int, optional
        Ending frame index (exclusive)
    roi : tuple, optional
        Region of interest (x_min, x_max, y_min, y_max) to crop
    
    Returns:
    --------
    None
    """
    # Handle stimulus_df input
    if isinstance(stimulus_df, dict):
        if worm_key is None:
            raise ValueError("worm_key must be specified when stimulus_df is a dict")
        if worm_key not in stimulus_df:
            raise ValueError(f"worm_key '{worm_key}' not found in stimulus_df")
        df = stimulus_df[worm_key]
    else:
        df = stimulus_df
    
    # Resolve folder and collect TIFF files (supports one level of subfolders)
    parent_folder = Path(parent_folder)
    if not parent_folder.exists():
        raise ValueError(f"Parent folder {parent_folder} does not exist")

    if parent_folder.is_file():
        raise ValueError(f"Expected a folder but received file path: {parent_folder}")

    tiff_files = list(parent_folder.glob(frame_pattern))

    subfolders = sorted((p for p in parent_folder.iterdir() if p.is_dir()), key=lambda p: p.name)
    for subfolder in subfolders:
        tiff_files.extend(subfolder.glob(frame_pattern))

    tiff_files = sorted(tiff_files)

    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {parent_folder} or its subfolders with pattern {frame_pattern}")
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Apply frame range if specified
    if start_frame is not None:
        tiff_files = tiff_files[start_frame:]
    if end_frame is not None:
        tiff_files = tiff_files[:end_frame - (start_frame or 0)]
    print(f"Processing {len(tiff_files)} frames")
    
    # Load stimulus mapping if provided
    stimulus_mapping = None
    if stimulus_mapping_file:
        if os.path.exists(stimulus_mapping_file):
            with open(stimulus_mapping_file, 'r', encoding='utf-8') as f:
                stimulus_mapping = json.load(f)
        else:
            print(f"Warning: Stimulus mapping file not found: {stimulus_mapping_file}")

    # Read first image to get dimensions
    first_img = tifffile.imread(str(tiff_files[0]))
    
    # Apply ROI if specified
    if roi is not None:
        x_min, x_max, y_min, y_max = roi
        first_img = first_img[y_min:y_max, x_min:x_max]

    if first_img.dtype != np.uint16:
        print(f"Warning: Image dtype is {first_img.dtype}, expected uint16")
    
    height, width = first_img.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(
        str(output_video_path), 
        fourcc, 
        fps, 
        (width, height),
        isColor=False  # Grayscale
    )
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video_path}")
    
    print(f"Creating video: {output_video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Codec: {codec}")
    
    # Process each frame
    frame_offset = start_frame or 0
    
    for idx, tiff_file in enumerate(tqdm(tiff_files, desc="Generating video")):
        # Read TIFF file
        img = tifffile.imread(str(tiff_file))
        
        # Apply ROI
        if roi is not None:
            x_min, x_max, y_min, y_max = roi
            img = img[y_min:y_max, x_min:x_max]
        
        # Normalize to uint8
        img_normalized = normalize_uint16_to_uint8(img, contrast_limits)
        
        # Get current state
        current_frame_idx = frame_offset + idx
        state = get_state_at_frame(current_frame_idx, df, stimulus_mapping)
        
        # Add text label
        img_with_text = add_text_to_frame(
            img_normalized,
            text=state,
            position=text_position,
            font_scale=font_scale,
            color=text_color,
            bg_color=text_bg_color
        )
        
        # Write frame
        out.write(img_with_text)
    
    # Release video writer
    out.release()
    
    print(f"Video saved to: {output_video_path}")


# Example usage
if __name__ == "__main__":
    from data_load.get_stimulus_info import get_stimulus_info
    
    parent_folder = r"I:\WJH\raw\20251105_bac\w5\MIP_Camera-Green_VSC-09321"
    output_video = r"I:\WJH\raw\20251105_bac\w5\output_video.avi"
    excel_path = r"H:\Process_temporary\WJH\olfactory\labjack_result\20251105_bac\output_volumes.xlsx"
    stimulus_json_path = r"h:\Process_temporary\WJH\sensory_pipeline_python\data\config\stimulus.json"
    
    # Get stimulus info
    stimulus_info = get_stimulus_info(excel_path)
    
    # Generate video
    tiff_to_video(
        parent_folder=parent_folder,
        output_video_path=output_video,
        stimulus_df=stimulus_info,
        worm_key='w5',  # Specify which worm if stimulus_info is a dict
        contrast_limits=(100, 305),
        fps=1,
        codec='MJPG', # can use MJPG for avi to load in ImageJ
        text_position='top_right',
        font_scale=1.0,
        stimulus_mapping_file=stimulus_json_path,
    )
