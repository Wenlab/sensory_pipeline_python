import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from dPCA import dPCA

class EpochdPCA:
    def __init__(self, neuron_segments_dict):
        """
        Initialize the EpochdPCA class with neuron segments data.
        """
        self.data = neuron_segments_dict
        self.dpca_results = None
        self.arranged_data = None
        self.dpca_all = None
        self.neuron_index_map = {}
        self.stimulus_index_map = {}

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

    def plot_dpca_results(self):
        """
        Plot the results of dPCA analysis.
        
        :return: Matplotlib figure object.
        """
        if self.dpca_results is None:
            raise ValueError("dPCA results not computed. Call perform_dpca() first.")
        Z = self.dpca_results
        time = np.arange(Z['t'].shape[2])
        plt.figure(figsize=(16, 7))

        plt.subplot(131)
        for s in range(Z['t'].shape[1]):
            plt.plot(time, Z['t'][0, s])
        plt.title('1st time component')

        plt.subplot(132)
        for s in range(Z['s'].shape[1]):
            plt.plot(time, Z['s'][0, s])
        plt.title('1st stimulus component')

        plt.subplot(133)
        for s in range(Z['st'].shape[1]):
            plt.plot(time, Z['st'][0, s])
        plt.title('1st mixing component')
        plt.tight_layout()
        plt.show()     


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.HDF5_load import load_h5file
    from result_analysis.analyse import run_analysis_pipeline
    neuron_segments_dict = load_h5file(
            path = r"I:\WJH\flavor\neuron_segments_dict_filter.h5",
            root_name= 'neuron_segments_dict')
    
    from result_analysis.baseline_correction import BaselineCorrection
    neuron_segments_dict_correct = BaselineCorrection(neuron_segments_dict)
    neuron_segments_dict_correct.apply_baseline_correction()
    neuron_segments_dict = neuron_segments_dict_correct.corrected_data
    dpca_analyser = EpochdPCA(neuron_segments_dict)
    dpca_analyser.arrange_data()
    dpca_analyser.fill_blank_neuron_stimuli_pair()
    dpca_analyser.perform_dpca()
    dpca_analyser.plot_dpca_results()
