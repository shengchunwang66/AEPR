# This is a script to generate spin parameters for the spin-trapping
import pandas as pd
import numpy as np
import itertools


# Read eprlib
data = pd.read_excel('SPIN_trapping_database_20250713_en.xlsx')

AN1 = list(np.round(data['AN1'].tolist(), 0))
AH1 = list(np.round(data['AH1'].tolist(), 0))
AH2 = list(np.round(data['AH2'].tolist(), 0))
AN2 = list(np.round(data['AN2'].tolist(), 0))
AH3 = list(np.round(data['AH3'].tolist(), 0))
AH4 = list(np.round(data['AH4'].tolist(), 0))


# Remove repeat data
tmp_list = []
for i in range(len(AN1)):
    tmp_list.append((AN1[i], AH1[i], AH2[i], AN2[i], AH3[i], AH4[i]))
tmp_list = list(set(tmp_list))

AN1, AH1, AH2, AN2, AH3, AH4 = [], [], [], [], [], []
for i in range(len(tmp_list)):
    AN1.append(tmp_list[i][0])
    AH1.append(tmp_list[i][1])
    AH2.append(tmp_list[i][2])
    AN2.append(tmp_list[i][3])
    AH3.append(tmp_list[i][4])
    AH4.append(tmp_list[i][5])
    

# genrate lwpp list
lwpp_values = np.round(np.arange(1.0, 4.1, 0.1), 1).tolist()


def generate_parameter_space(original_params, lwpp_values, step=[0.125, 0.125, 0.125, 0.25, 0.25, 0.25], range_offset=[0.5, 0.5, 0.5, 1, 1, 1]):
    """
    Generate parameters combinations
    
    Args:
        original_params: original para list [AN1, AH1, AH2, AN2, AH3, AH4]
        lwpp_values: lwpp list
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
    
    # add lwpp
    param_ranges.append(lwpp_values)
    
    # generate and remove
    combinations = list(itertools.product(*param_ranges))
    combinations = list(set(combinations))

    return combinations

def process_all_combinations():
    """
    process all combinations
    """
    all_new_combinations = []
    
    original_combinations = list(zip(AN1, AH1, AH2, AN2, AH3, AH4))
    
    for i, original_combo in enumerate(original_combinations):
        if i % 100 == 0:
            print(f"Processing: {i}/{len(original_combinations)}")
        
        # generate
        new_combinations = generate_parameter_space(original_combo, lwpp_values)
        all_new_combinations.extend(new_combinations)
    
    print(f"Generate {len(all_new_combinations)} new combinations in total.")
    return all_new_combinations


# obtain new parameters
new_combinations = process_all_combinations()


# Save the results
def save_combinations_to_file(combinations, filename="generated_spin.txt"):
    df = pd.DataFrame(combinations, columns=['AN1', 'AH1', 'AH2', 'AN2', 'AH3', 'AH4', 'lwpp'])
    df.to_csv(filename, index=False, sep=' ', header=False)


save_combinations_to_file(new_combinations)

