import numpy as np
import scipy.stats as stats
from dPCA import dPCA
import hashlib
import json
import os
from scipy.stats import sem
import sys
if __name__ == "__main__":
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parse_stimulus_info import group_and_sort_stimuli

class SegmentdPCA:
    def __init__(self, neuron_segments_dict, compound_info=None, compound_color_scheme=None):
        """
        Initialize the SegmentdPCA class with neuron segments data.
        
        Parameters:
        -----------
        neuron_segments_dict : dict
            Dictionary with structure {neuron_group: {stimulus_type: [trial data]}}
        compound_info : dict, optional
            Maps stimulus codes to descriptive names. If None, will extract from trial 'stim_name'.
        compound_color_scheme : dict, optional
            Maps stimulus codes to colors. If None, will extract from trial 'stim_color'.
        """
        self.data = neuron_segments_dict
        self.dpca_results = None
        self.arranged_data = None
        self.dpca_all = None
        self.neuron_index_map = {}
        self.stimulus_index_map = {}
        
        # Extract metadata from embedded trial data if not provided
        if compound_info is None or compound_color_scheme is None:
            extracted_info, extracted_colors = self._extract_metadata_from_trials()
            if compound_info is None:
                compound_info = extracted_info
            if compound_color_scheme is None:
                compound_color_scheme = extracted_colors
        
        self.compound_info = compound_info
        self.compound_color_scheme = compound_color_scheme
    
    def _extract_metadata_from_trials(self):
        """Extract stim_name and stim_color from embedded trial data."""
        compound_info = {}
        compound_color_scheme = {}
        
        for neuron, stimuli in self.data.items():
            for stim_type, segments in stimuli.items():
                if segments:
                    first_seg = segments[0]
                    if stim_type not in compound_info:
                        compound_info[stim_type] = first_seg.get('stim_name', stim_type)
                    if stim_type not in compound_color_scheme:
                        compound_color_scheme[stim_type] = first_seg.get('stim_color', '#808080')
        
        return compound_info, compound_color_scheme

    def get_neuron_stimuli_info(self):
        """
        Get the list of neuron names and stimuli from the data.
        
        :return: Tuple of (neuron_names, stimuli)
        """
        neuron_names = list(self.data.keys())
        stimuli = set()
        for neuron in neuron_names:
            stimuli.update(self.data[neuron].keys())
        return neuron_names, list(stimuli)

    def arrange_data(self):
        """
        Arrange data for dPCA analysis.
        
        :return: Averaged Trials in format (Neurons, Stimulus, Time).
        """
        neuron_names, stimuli = self.get_neuron_stimuli_info()
        num_neurons = len(neuron_names)
        num_stimuli = len(stimuli)

        # Assuming all trials have the same length
        time_points = len(self.data[neuron_names[0]][stimuli[0]][0]['deltaFoverF_0'])

        # Create mapping from names to indices
        neuron_index_map = {name: idx for idx, name in enumerate(neuron_names)}
        stimulus_index_map = {stimulus: idx for idx, stimulus in enumerate(stimuli)}

        # Initialize the trials array
        trials = np.zeros((num_neurons, num_stimuli, time_points))

        # Loop through neurons and stimuli to fill the trials array
        for neuron_name, stimuli_data in self.data.items():
            neuron_index = neuron_index_map[neuron_name]
            for stimulus, trials_data in stimuli_data.items():
                stimulus_idx = stimulus_index_map[stimulus]

                all_trials_for_condition = []
                # Collect trials for this specific stimulus
                for trial in trials_data:
                    all_trials_for_condition.append(trial['deltaFoverF_0'])

                if all_trials_for_condition:
                    # Calculate mean across trials for this neuron-stimulus pair
                    trials[neuron_index, stimulus_idx, :] = np.mean(all_trials_for_condition, axis=0)
        self.arranged_data = trials
        self.neuron_index_map = neuron_index_map
        self.stimulus_index_map = stimulus_index_map

    def fill_blank_neuron_stimuli_pair(self):
        """
        Fill in blank neuron-stimuli pair with neuron that is symmetric in anatomy, for example, AWCL and AWCR.
        """
        # find if there is a neuron_stimuli_pair has all zeros
        for neuron_idx in range(self.arranged_data.shape[0]):
            for stimulus_idx in range(self.arranged_data.shape[1]):
                if np.all(self.arranged_data[neuron_idx, stimulus_idx, :] == 0):
                    neuron_name = list(self.neuron_index_map.keys())[neuron_idx]
                    symmetric_neuron_index = self._find_symmetric_neuron_index(neuron_name)
                    if symmetric_neuron_index is not None:
                        self.arranged_data[neuron_idx, stimulus_idx, :] = self.arranged_data[symmetric_neuron_index, stimulus_idx, :]

    def _find_symmetric_neuron_index(self, neuron_name):
        """
        Find the index of the neuron that is symmetric to the given neuron.
        
        :param neuron_name: Name of the neuron.
        :return: Index of the symmetric neuron or None if not found.
        """
        if neuron_name.endswith('L'):
            symmetric_name = neuron_name[:-1] + 'R'
        elif neuron_name.endswith('R'):
            symmetric_name = neuron_name[:-1] + 'L'
        else:
            return None
        
        return self.neuron_index_map.get(symmetric_name, None)
   
    def perform_dpca(self, n_components=10):
        """
        Perform dPCA on the arranged data.
        
        :param n_components: Number of components to compute.
        :return: dPCA results.
        """        
        dpca = dPCA.dPCA(labels='st', n_components=n_components)
        dpca.protect = ['t']
        Z = dpca.fit_transform(self.arranged_data)
        self.dpca_results = Z
        self.dpca_all = dpca
        return self.dpca_results
    
    def get_compound_summary(self):
        """
        Print a summary of compounds and their concentrations in the dataset.
        """
        print("Compound Summary:")
        print("=" * 50)
        
        # Group by compound type
        compound_groups = {}
        for stimulus in self.stimulus_index_map.keys():
            if '_' in stimulus:
                base_name = stimulus.split('_')[0]
                if base_name not in compound_groups:
                    compound_groups[base_name] = []
                compound_groups[base_name].append(stimulus)
        
        for base_name, stimuli in sorted(compound_groups.items()):
            # Get the compound name from the first stimulus
            first_stimulus = stimuli[0]
            if self.compound_info and first_stimulus in self.compound_info:
                compound_type = self.compound_info[first_stimulus].split()[0]
                print(f"\n{base_name.upper()} ({compound_type}):")
            else:
                print(f"\n{base_name.upper()}:")
            
            stimuli.sort(key=lambda x: int(x.split('_')[1]) if len(x.split('_')) > 1 else 0)
            for stimulus in stimuli:
                compound_name = self.compound_info.get(stimulus, stimulus) if self.compound_info else stimulus
                print(f"  - {stimulus}: {compound_name}")