# This is a script to generate metal parameters
import pandas as pd
import numpy as np
import itertools


# Read eprlib
data = pd.read_excel('WSC-EPR_metal01222026.xlsx')

A1 = list(np.round(data['A1'].tolist(), 0))
A2 = list(np.round(data['A2'].tolist(), 0))
A3 = list(np.round(data['A3'].tolist(), 0))
G1 = list(np.round(data['G1'].tolist(), 0))
G2 = list(np.round(data['G2'].tolist(), 0))
G3 = list(np.round(data['G3'].tolist(), 0))
lwpp1 = list(np.round(data['lwpp1'].tolist(), 0))
lwpp2 = list(np.round(data['lwpp2'].tolist(), 0))
lwpp3 = list(np.round(data['lwpp3'].tolist(), 0))


# Remove repeat data
tmp_list = []
for i in range(len(A1)):
    tmp_list.append((A1[i], A2[i], A3[i], G1[i], G2[i], G3[i], lwpp1[i], lwpp2[i], lwpp3[i]))
tmp_list = list(set(tmp_list))

A1, A2, A3, G1, G2, G3, lwpp1, lwpp2, lwpp3 = [], [], [], [], [], [], [], [], []
for i in range(len(tmp_list)):
    A1.append(tmp_list[i][0])
    A2.append(tmp_list[i][1])
    A3.append(tmp_list[i][2])
    G1.append(tmp_list[i][3])
    G2.append(tmp_list[i][4])
    G3.append(tmp_list[i][5])
    lwpp1.append(tmp_list[i][6])
    lwpp2.append(tmp_list[i][7])
    lwpp3.append(tmp_list[i][8])



def generate_parameter_space(original_params, step=[0.125, 0.125, 0.125, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125], range_offset=[0.5, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.5]):
    """
    Generate parameters combinations
    
    Args:
        original_params: original para list [A1, A2, A3, G1, G2, G3, lwpp1, lwpp2, lwpp3]
        step: default 0.125 and 0.25
        range_offset: default 0.5 and 1.0
    
    Returns:
        all generated combinations
    """
        
    param_ranges = []
    
    for index, param in enumerate(original_params):
        # if para is 0, keep unchanged
        if param == 0:
            param_range = [0.0]
        else:
            start = param - range_offset[index]
            end = param + range_offset[index] + step[index]
            param_range = np.round(np.arange(start, end, step[index]), 2)
        
        param_ranges.append(param_range)
    
    # generate and remove
    combinations = list(itertools.product(*param_ranges))
    combinations = list(set(combinations))

    return combinations

def process_all_combinations():
    """
    process all combinations
    """
    all_new_combinations = []
    
    original_combinations = list(zip(A1, A2, A3, G1, G2, G3, lwpp1, lwpp2, lwpp3))
    
    for i, original_combo in enumerate(original_combinations):
        if i % 100 == 0:
            print(f"Processing: {i}/{len(original_combinations)}")
        
        # generate
        new_combinations = generate_parameter_space(original_combo)
        all_new_combinations.extend(new_combinations)
    
    print(f"Generate {len(all_new_combinations)} new combinations in total.")
    return all_new_combinations


# obtain new parameters
new_combinations = process_all_combinations()


# Save the results
def save_combinations_to_file(combinations, filename="generated_metal.txt"):
    df = pd.DataFrame(combinations, columns=['A1', 'A2', 'A3', 'G1', 'G2', 'G3', 'lwpp1', 'lwpp2', 'lwpp3'])
    df.to_csv(filename, index=False, sep=' ', header=False)


save_combinations_to_file(new_combinations)

