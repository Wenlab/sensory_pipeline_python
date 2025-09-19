def get_symmetric_neuron(neuron_name):
    if isinstance(neuron_name, str):
        if neuron_name.endswith('L'):
            return neuron_name[:-1] + 'R'
        elif neuron_name.endswith('R'):
            return neuron_name[:-1] + 'L'
        else:
            return None