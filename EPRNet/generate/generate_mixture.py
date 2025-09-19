# This code is used to generate the mixture of two or three compounds
import pandas as pd
import numpy as np
import itertools


# Iterive lwpp list
lwpp_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Read eprlib
data = pd.read_excel('SPIN_trapping_database_20250713_en.xlsx')

AN1 = list(np.round(data['AN1'].tolist(), 0))
AH1 = list(np.round(data['AH1'].tolist(), 0))
AH2 = list(np.round(data['AH2'].tolist(), 0))
AN2 = list(np.round(data['AN2'].tolist(), 0))
AH3 = list(np.round(data['AH3'].tolist(), 0))
AH4 = list(np.round(data['AH4'].tolist(), 0))
buhuo = data['Trap reagents'].tolist()
radical = data['Radical type'].tolist()

for i in range(len(buhuo)):
    buhuo[i] = buhuo[i][buhuo[i].rfind('(')+1:buhuo[i].rfind(')')].upper()
    
buhuo_set = list(set(buhuo))


# Remove repeat data
tmp_list = []
paras, buhuo_new, radical_new = [], [], []
for i in range(len(AN1)):
    if (AN1[i], AH1[i], AH2[i], AN2[i], AH3[i], AH4[i]) not in tmp_list:
        tmp_list.append((AN1[i], AH1[i], AH2[i], AN2[i], AH3[i], AH4[i]))
        paras.append((AN1[i], AH1[i], AH2[i], AN2[i], AH3[i], AH4[i]))
        buhuo_new.append(buhuo[i])
        radical_new.append(radical[i])


combs_mixture = []

# *** three compounds *** #
# mixture ratio list
ratio_list = [(0.3, 0.3, 0.4)]
for i in range(len(buhuo_set)):
    index_b = [i_b for i_b, x in enumerate(buhuo_new) if x == buhuo_set[i]]
    
    combs_valid = []
    for j in range(len(index_b)-2):
        for k in range(j+1, len(index_b)-1):
            for m in range(k+1, len(index_b)):
                if radical_new[j] != radical_new[k] and radical_new[j] != radical_new[m] and radical_new[k] != radical_new[m]:
                    combs_valid.append([j, k, m])

            
    for lwpp in lwpp_list:     
        for ratio in ratio_list:
            for combs in combs_valid:
                combs_mixture.append([combs, ratio, lwpp])   
        
        
# *** two compounds *** #
# mixture ratio list
ratio_list = [(0.5, 0.5)]
for i in range(len(buhuo_set)):
    index_b = [i_b for i_b, x in enumerate(buhuo_new) if x == buhuo_set[i]]
    
    combs_valid = []
    for j in range(len(index_b)-1):
        for k in range(j+1, len(index_b)):
            if radical_new[j] != radical_new[k]:
                combs_valid.append([j, k])
            
    for lwpp in lwpp_list:     
        for ratio in ratio_list:
            for combs in combs_valid:
                comb_paras = [paras[i] for i in combs]
                combs_mixture.append([comb_paras, ratio, lwpp])               
    
    
print(len(combs_mixture))


# Save the results
def save_combinations_to_txt(combinations, filename="generated_mixture.csv"):
    df = pd.DataFrame(combinations, columns=['comb_paras', 'ratio', 'lwpp'])
    df.to_csv(filename, index=False)


save_combinations_to_txt(combs_mixture)

