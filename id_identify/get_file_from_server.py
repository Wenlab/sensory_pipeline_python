import paramiko
import os
import re
from tqdm import tqdm
import stat
from pathlib import Path
import numpy as np
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.read_vols_using_dask import lazy_read_tiff_stack

def save_dask_array_as_npy(dask_array, output_path):
    # Convert the Dask array to a NumPy array
    numpy_array = dask_array.compute()
    # Ensure the output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the NumPy array to a .npy file
    np.save(output_path, numpy_array)

def get_ex_ref_image(folder_path, output_path, t_start=0, t_end=2, **kwargs):
    # Get all subfolders in the directory
    subfolders = [f for f in Path(folder_path).iterdir() if f.is_dir()]
    processed_prefixes = set()
    for subfolder in subfolders:
        prefix = subfolder.name.split("_")[0]
        if prefix not in processed_prefixes:
            processed_prefixes.add(prefix)
            # Process each subfolder
            # if subfolder startswith("w"):
            if subfolder.name.startswith("w"):
                # Process the subfolder
                print(f"Processing {subfolder}...")
                tiff_dir_name = os.path.join(folder_path, subfolder.name)
                exp_path = tiff_dir_name
                red_tiff_path_ = rf"{exp_path}\0_Camera-Red_VSC-10629"
                green_tiff_path_ = rf"{exp_path}\1_Camera-Green_VSC-09321"
                volume_read_params = dict(
                    z_start_frame_number=kwargs.get("z_start_frame_number",0),
                    z_end_frame_number=kwargs.get("z_end_frame_number",17),
                    mod2_reverse=[False, False],
                    img_width=kwargs.get("img_width",1024),
                    img_height=kwargs.get("img_height",1024),
                    frame_number_per_volume=kwargs.get("frame_number_per_volume",20),
                    img_dtype=np.uint16,
                )
                red = lazy_read_tiff_stack(red_tiff_path_, volume_read_params)
                green = lazy_read_tiff_stack(green_tiff_path_, volume_read_params)
                save_dask_array_as_npy(red[t_start:t_end], os.path.join(output_path, subfolder.name.split("_")[0], "red.npy"))
                save_dask_array_as_npy(green[t_start:t_end], os.path.join(output_path, subfolder.name.split("_")[0], "green.npy"))
            else:
                if subfolder.name.startswith("W"):
                    print(f"Processing Ref Volumes in {subfolder}...")
                    tiff_dir_name = os.path.join(folder_path, subfolder.name)
                    exp_path = tiff_dir_name
                    red_tiff_path_ = rf"{exp_path}\0_Camera-Red_VSC-10629"
                    green_tiff_path_ = rf"{exp_path}\1_Camera-Green_VSC-09321"
                    volume_read_params = dict(
                        z_start_frame_number=kwargs.get("z_start_frame_number",0),
                        z_end_frame_number=kwargs.get("z_end_frame_number",17),
                        mod2_reverse=[False, False],
                        img_width=kwargs.get("img_width",1024),
                        img_height=kwargs.get("img_height",1024),
                        frame_number_per_volume=kwargs.get("frame_number_per_volume",20),
                        img_dtype=np.uint16,
                    )
                    red = lazy_read_tiff_stack(red_tiff_path_, volume_read_params)
                    green = lazy_read_tiff_stack(green_tiff_path_, volume_read_params)
                    save_dask_array_as_npy(red[0], os.path.join(output_path, "ref_red.npy"))
                    save_dask_array_as_npy(green[0], os.path.join(output_path, "ref_green.npy"))
                else:
                    print(f"Skipping {subfolder}, does not match expected naming convention.")



def establish_ssh_connection(hostname, port, username, password=None, key_filename=None):
    """
    Establishes SSH connection to the remote server.
    
    Args:
        hostname (str): The hostname or IP address of the remote server.
        port (int): The SSH port number.
        username (str): The username for SSH authentication.
        password (str, optional): The password for SSH authentication.
        key_filename (str, optional): Path to private key file for key-based authentication.
    
    Returns:
        tuple: (ssh_client, sftp_client) or (None, None) if connection fails.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        if key_filename:
            ssh.connect(hostname, port=port, username=username, key_filename=key_filename)
        else:
            ssh.connect(hostname, port=port, username=username, password=password)
        
        sftp = ssh.open_sftp()
        print(f"Successfully connected to {hostname}:{port}")
        return ssh, sftp
    
    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your credentials.")
        return None, None
    except paramiko.SSHException as e:
        print(f"Could not establish SSH connection: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        return None, None

def discover_worm_folders(sftp, remote_base_path):
    """
    Automatically discovers all 'w{i}' folders in the remote base path.
    
    Args:
        sftp: SFTP client object.
        remote_base_path (str): The base path on the remote server.
    
    Returns:
        list: List of worm numbers found (e.g., [1, 2, 3, 5]).
    """
    try:
        files_and_dirs = sftp.listdir(remote_base_path)
        worm_pattern = re.compile(r'^w(\d+)$')
        worm_numbers = []
        
        for item in files_and_dirs:
            match = worm_pattern.match(item)
            if match:
                worm_numbers.append(int(match.group(1)))
        
        worm_numbers.sort()
        print(f"Discovered worm folders: {[f'w{num}' for num in worm_numbers]}")
        return worm_numbers
    
    except Exception as e:
        print(f"Error discovering worm folders: {e}")
        return []

def check_and_remove_useless_worms(sftp, remote_base_path, worm_list):
    """
    Checks for 'useless' marker files and removes corresponding worms from the list.
    
    Args:
        sftp: SFTP client object.
        remote_base_path (str): The base path on the remote server.
        worm_list (list): List of worm numbers to check.
    
    Returns:
        list: Filtered list of worm numbers without useless worms.
    """
    useful_worms = []
    
    for worm_num in worm_list:
        worm_folder = f"w{worm_num}"
        useless_file_path = os.path.join(remote_base_path, worm_folder, "useless").replace('\\', '/')
        
        try:
            # Try to access the useless file
            sftp.stat(useless_file_path)
            print(f"Skipping {worm_folder}: marked as useless")
        except FileNotFoundError:
            # No useless file found, worm is useful
            useful_worms.append(worm_num)
        except Exception as e:
            print(f"Error checking useless file for {worm_folder}: {e}")
            # If we can't check, assume it's useful
            useful_worms.append(worm_num)
    
    print(f"Useful worms after filtering: {[f'w{num}' for num in useful_worms]}")
    return useful_worms

def download_directory(sftp, remote_dir, local_dir):
    """
    Downloads an entire directory from the remote server to the local machine.
    
    Args:
        sftp: SFTP client object.
        remote_dir (str): The remote directory to download.
        local_dir (str): The local directory to download.
    """
    os.makedirs(local_dir, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        remote_item_path = os.path.join(remote_dir, item.filename).replace('\\', '/')
        local_item_path = os.path.join(local_dir, item.filename)
        
        if stat.S_ISDIR(item.st_mode):
            # It's a directory, recurse
            download_directory(sftp, remote_item_path, local_item_path)
        else:
            # It's a file, download it
            sftp.get(remote_item_path, local_item_path)

def download_files_for_worm(sftp, remote_base_path, local_base_path, worm_num, items_to_download):
    """
    Downloads specified files and directories for a single worm.
    
    Args:
        sftp: SFTP client object.
        remote_base_path (str): The base path on the remote server.
        local_base_path (str): The base path on the local machine.
        worm_num (int): The worm number.
        items_to_download (list): List of file or directory paths to download.
    """
    remote_w_folder = os.path.join(remote_base_path, f"w{worm_num}").replace('\\', '/')
    local_w_folder = os.path.join(local_base_path, f"w{worm_num}")
    
    # Ensure local directory exists
    os.makedirs(local_w_folder, exist_ok=True)
    
    for item_path in items_to_download:
        remote_item_path = os.path.join(remote_w_folder, item_path).replace('\\', '/')
        local_item_path = os.path.join(local_w_folder, item_path)
        
        try:
            item_stat = sftp.stat(remote_item_path)
            
            if stat.S_ISDIR(item_stat.st_mode):
                # It's a directory
                print(f"Downloading directory: {remote_item_path}")
                download_directory(sftp, remote_item_path, local_item_path)
            else:
                # It's a file
                print(f"Downloading file: {remote_item_path}")
                # Ensure parent directory exists locally
                os.makedirs(os.path.dirname(local_item_path), exist_ok=True)
                sftp.get(remote_item_path, local_item_path)

        except FileNotFoundError:
            print(f"Error: Remote item not found - {remote_item_path}")
        except Exception as e:
            print(f"Error processing {remote_item_path}: {e}")


def download_files_from_server(
    hostname,
    username,
    password=None,
    key_filename=None,
    port=22,
    remote_base_path=None,
    local_base_path=None,
    worm_list=None,
    files_to_download=None
):
    """
    Downloads specific files from 'w{i}' folders on a remote server
    to corresponding local 'w{i}' folders.

    Args:
        hostname (str): The hostname or IP address of the remote server.
        username (str): The username for SSH authentication.
        password (str, optional): The password for SSH authentication.
        key_filename (str, optional): Path to private key file for key-based authentication.
        port (int): The SSH port number (default: 22).
        remote_base_path (str): The base path on the remote server where 'w{i}' folders reside.
        local_base_path (str): The base path on your local machine where 'w{i}' folders will be created.
        worm_list (list, optional): List of worm numbers to download. If None, auto-discovers all worms.
        files_to_download (list, optional): List of file paths to download from each worm folder.
    """
    if files_to_download is None:
        files_to_download = [
            # "synthetic_volume/aligned_volumes_mip.npy",
            # "synthetic_volume/all_neuron_pt_tuple.npy"
            "output/synthetic_mip_results/aligned_volumes_mip.npy",
            "output/synthetic_mip_results/neuron_pt_tuple.npy",
            "output/ex_neuron_pt_tuple.npy",
            "output/reference_inference_results/ref_neuron_pt_tuple_filled.npy"
        ]
    
    # Establish connection
    ssh, sftp = establish_ssh_connection(hostname, port, username, password, key_filename)
    if not ssh or not sftp:
        return
    
    try:
        # Auto-discover worms if worm_list is None
        if worm_list is None:
            worm_list = discover_worm_folders(sftp, remote_base_path)
            if not worm_list:
                print("No worm folders found.")
                return
        
        # Filter out useless worms
        useful_worms = check_and_remove_useless_worms(sftp, remote_base_path, worm_list)
        
        if not useful_worms:
            print("No useful worms found after filtering.")
            return
        
        # Download files for each useful worm
        for worm_num in tqdm(useful_worms):
            download_files_for_worm(sftp, remote_base_path, local_base_path, worm_num, files_to_download)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if ssh:
            ssh.close()
            print("\nSSH connection closed.")


# --- How to use it ---
if __name__ == "__main__":
    SERVER_HOSTNAME = "192.168.1.93"
    SERVER_PORT = 23
    SERVER_USERNAME = "wangjinghao"
    SERVER_PASSWORD = None  # Highly recommend using SSH keys instead of passwords
    # If using key-based authentication:
    # SSH_PRIVATE_KEY_PATH = "/path/to/your/private/key"

    REMOTE_BASE_PATH = "/home/data4/WJH/olfactory_result/20240901_wen0065"
    LOCAL_BASE_PATH = "I:/WJH/try"

    # Option 1: Specify specific worms
    # WORM_LIST = [1, 2, 3, 5]
    
    # Option 2: Auto-discover all worms (set to None)
    WORM_LIST = None

    download_files_from_server(
        hostname=SERVER_HOSTNAME,
        username=SERVER_USERNAME,
        password=SERVER_PASSWORD,
        # key_filename=SSH_PRIVATE_KEY_PATH, # Uncomment if using key-based auth
        port=SERVER_PORT,
        remote_base_path=REMOTE_BASE_PATH,
        local_base_path=LOCAL_BASE_PATH,
        worm_list=WORM_LIST
    )