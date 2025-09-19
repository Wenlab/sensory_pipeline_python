import re

def parse_concentration(concentration_str):
    '''
    Parse concentration string and return a tuple for sorting.
    '''
    if '%' in concentration_str:
        match = re.search(r'([\d.]+)%', concentration_str)
        if match:
            value = float(match.group(1))
            return (0, value)  # '%' concentrations come first
    elif 'uM' in concentration_str:
        match = re.search(r'([\d.]+)uM', concentration_str)
        if match:
            value = float(match.group(1))
            return (1, value)  # 'uM' concentrations come second
    elif 'mM' in concentration_str:
        match = re.search(r'([\d.]+)mM', concentration_str)
        if match:
            value = float(match.group(1))
            return (2, value)  # 'mM' concentrations come last
    
    elif 'E' in concentration_str:
        match = re.search(r'E(\d+)', concentration_str)
        if match:
            return (3, -float(match.group(1)))  # Negative for reverse sorting
    return (4, 0)

def group_and_sort_stimuli(stimulus_info_dict):
    compound_groups = {}
    for code, full_name in stimulus_info_dict.items():
        if ' E' in full_name:
            compound_name = full_name.split(' E')[0]
        else:
            parts = full_name.split()
            if len(parts) > 1:
                compound_name = ' '.join(parts[:-1])
            else:
                compound_name = full_name

        if compound_name not in compound_groups:
            compound_groups[compound_name] = []
        compound_groups[compound_name].append((code, full_name))
    
    sorted_groups = []
    for compound_name, stimuli in compound_groups.items():
        sorted_stimuli = sorted(stimuli, key=lambda x: parse_concentration(x[1]))
        stimulus_codes = [code for code, _ in sorted_stimuli]
        sorted_groups.append((compound_name, stimulus_codes))
    
    sorted_groups = sorted(sorted_groups, key=lambda x: x[0])
    return sorted_groups
