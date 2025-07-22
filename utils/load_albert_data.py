import scipy.io as sio
import numpy as np
from collections import defaultdict
import numpy as np
from datetime import datetime
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.interpolate import interpolate_over_nans
#%%
def load_albert_data(file_path, compound_list=None, dilution_factor="E4"):
    """
    Load and process Albert's(Sci.Advances) data from a .mat file.
    Parameters:
    -----------
    file_path : str
        Path to the .mat file containing Albert's data.
    compound_list : list, optional
        List of compounds to extract from the data. If None, all compounds will be extracted.
    dilution_factor : str, optional
        Dilution factor to use for the data. Default is "E4".
    """
    conc_dict = sio.loadmat(file_path)
    neuron_list = [str(item[0]) for item in conc_dict['neuron_pairs'][0]] 
    odor_list = conc_dict['odor_list']
    compile_keys = [key for key in conc_dict if key.startswith('compile')]
    compile_response = conc_dict[compile_keys[0]]
    all_neuron_trace_dict = defaultdict()
    if compound_list is not None:
        for compound in compound_list:
            if compound not in odor_list:
                raise ValueError(f"Compound '{compound}' not found in the data.")
            compound_index = np.where(odor_list == compound)[0][0]
            compile_response_compound = compile_response[0, compound_index]
            compile_response_compound = compile_response_compound.squeeze()
            neuron_trace_dict_compound = {}
            for i, neuron_struct in enumerate(compile_response_compound):
                if i >= len(neuron_list):
                    continue
                neuron_trace_dict_compound[neuron_list[i]] = neuron_struct['all_traces']
            # Interpolate over NaNs for each neuron trace
            for key, neuron_trace in neuron_trace_dict_compound.items():
                neuron_trace_dict_compound[key], _ = interpolate_over_nans(neuron_trace)
            all_neuron_trace_dict[str(compound)+f"_{dilution_factor}"] = neuron_trace_dict_compound
    else:
        compound_list = [str(item[0][0]) for item in odor_list]
        all_neuron_trace_dict = load_albert_data(file_path, compound_list=compound_list, dilution_factor=dilution_factor)
    return all_neuron_trace_dict


def transform_dict_for_dash(input_dict, start_time=1, end_time=26, date=None):
    """
    Transform dictionary from stimulus-organized format to neuron-organized format for Dash.
    
    Parameters:
    -----------
    input_dict : dict
        Dictionary with format {
            "stimulus_name": {
                'neuron_name': (Trials, Time) numpy array,
                ...
            },
            ...
        }
    start_time : float, optional
        Start time of stimulus (default: 0.0)
    end_time : float, optional
        End time of stimulus (default: 10.0)
    date : str, optional
        Date string. If None, uses current date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    dict
        Dictionary in format required by Dash:
        {
            'neuron_name': {
                "stimulus_name": [
                    {
                        'date': str,
                        'delta_FoverF_0': (T,) numpy array,
                        'end_time': float,
                        'start_time': float,
                        'segment_index': int,
                        'worm_key': str
                    },
                    ... # one dict per trial
                ]
            }
        }
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    output_dict = {}
    
    # First, collect all unique neuron names across all stimuli
    all_neurons = set()
    for stimulus_name, neuron_data in input_dict.items():
        all_neurons.update(neuron_data.keys())
    
    # Initialize the output structure for each neuron
    for neuron_name in all_neurons:
        output_dict[neuron_name] = {}
    
    # Process each stimulus
    for stimulus_name, neuron_data in input_dict.items():
        for neuron_name, trials_data in neuron_data.items():
            # trials_data shape: (Trials, Time)
            num_trials, num_timepoints = trials_data.shape
            
            # Initialize stimulus entry for this neuron as a list
            if stimulus_name not in output_dict[neuron_name]:
                output_dict[neuron_name][stimulus_name] = []
            
            # Create entries for each trial as list elements
            for trial_idx in range(num_trials):
                trial_dict = {
                    'date': date,
                    'deltaFoverF_0': trials_data[trial_idx, :],  # (T,) array for single trial
                    'end_time': end_time,
                    'start_time': start_time,
                    'segment_index': trial_idx,
                    'worm_key': 'w1'
                }
                output_dict[neuron_name][stimulus_name].append(trial_dict)
    
    return output_dict

def load_and_transform_albert_data(file_path, compound_list=None, dilution_factor="E4", start_time=1, end_time=26, date=None):
    """
    Load and transform Albert's data from a .mat file.

    Parameters:
    -----------
    file_path : str
        Path to the .mat file containing Albert's data.
    compound_list : list, optional
        List of compounds to extract from the data. If None, all compounds will be extracted.
    dilution_factor : str, optional
        Dilution factor to use for the data. Default is "E4".
    start_time : float, optional
        Start time of stimulus (default: 1.0)
    end_time : float, optional
        End time of stimulus (default: 26.0)
    date : str, optional
        Date string. If None, uses current date in 'YYYY-MM-DD' format

    Returns:
    --------
    dict
        Transformed data in the format required by Dash.
    """
    # Load the data
    all_neuron_trace_dict = load_albert_data(file_path, compound_list=compound_list, dilution_factor=dilution_factor)

    # Transform the data
    transformed_data = transform_dict_for_dash(all_neuron_trace_dict, start_time, end_time, date)

    return transformed_data

if __name__ == "__main__":
    compound_list_1 = ['1-octanol', 'geraniol', 'benzaldehyde']
    transformed_data_E4 = load_and_transform_albert_data(r"I:\WJH\flavor\Albert_data\CElegans Chemosensory Data\Compiled Neural Data\compiled_data_conc_4.mat", dilution_factor="4")