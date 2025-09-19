import os
import numpy as np
import pandas as pd
from scipy import interpolate


# read prediction
def read_soph_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        temp = []
        for line in lines:
            tt = line.split('\t')
            try:
                for i in range(len(tt)):
                    tt[i] = round(float(tt[i]), 5)
                temp.append(tt)
            
            except Exception as e:
                continue
            
    return temp


# read original sepctra
def read_origin_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        temp = []
        for line in lines:
            line = line.replace('\n', '')
            try:
                if ',' in line:
                    tt = line.split(',')
                    for i in range(len(tt)):
                        tt[i] = round(float(tt[i]), 5)
                    temp.append(tt)
                elif '\t' in line:
                    tt = line.split('\t')
                    for i in range(len(tt)):
                        tt[i] = round(float(tt[i]), 5)
                    temp.append(tt)
                else:
                    new_tt = []
                    tt = line.split(' ')
                    for i in range(len(tt)):
                        if len(tt[i]) > 0:
                            new_tt.append(round(float(tt[i]), 5))
                    temp.append(new_tt)
            except Exception as e:
                continue
    return temp


# Auto peak detection
def find_peaks_shift(currentVector, thre=5e-5):
    currentVector = currentVector[0, :]
    cv_posorg = np.where(abs(currentVector)-thre > 0, currentVector-thre, 0)

    cv_pos_new_left = np.roll(cv_posorg, -1)
    cv_pos_new_right = np.roll(cv_posorg, 1)
        
    cv_diff_left = cv_posorg - cv_pos_new_left
    cv_diff_right = cv_posorg - cv_pos_new_right

    cv_pos_l = np.sign(cv_diff_left)
    cv_pos_r = np.sign(cv_diff_right)

    cv_pos_sum = cv_pos_l + cv_pos_r

    peaks_cv = np.where(cv_pos_sum==2)
        
    return peaks_cv[0]


# calculate the correct ratio
def calculate_ratio_below_threshold(all_avg_dis_x, all_avg_dis_xy, threshold):
    # check length
    if len(all_avg_dis_x) != len(all_avg_dis_xy):
        raise ValueError("different length！")
    
    # stats
    count = 0
    for x, xy in zip(all_avg_dis_x, all_avg_dis_xy):
        if x < threshold and xy < threshold:
            count += 1
    
    total = len(all_avg_dis_x)
    ratio = count / total if total > 0 else 0.0
    
    return ratio


if __name__ = "__main__":
    ### for mixture
    root_path = 'D:/2025/ailab/public/final_0716/mixture/'

    # ### for spin-trapping
    # root_path = 'D:/2025/ailab/public/final_0716/spin_trapping/'
    threshold = 0.5

    org_peak_path = os.path.join(root_path, 'peaks_file')
    org_data_path = os.path.join(root_path, 'data')
    res_path = os.path.join(root_path, 'results')
    org_peak = []
    org_data = []
    names = []

    for item in sorted(os.listdir(org_peak_path)):
        names.append(int(item.replace('.txt', '')))
        with open(os.path.join(org_peak_path, item),'r') as f:
            lines = f.readlines()
            temp = []
            for line in lines:
                temp.append(int(line.replace('\n', '')))
            org_peak.append(temp)

    for item in sorted(os.listdir(org_data_path)):
        data = read_origin_txt(os.path.join(org_data_path, item))
        
        data = np.array(data)
        org_data.append(data)



    cnt = 0
    fit_peaks = []
    fit_data = []
    for item in sorted(os.listdir(res_path)):
        item_path = os.path.join(res_path, item, 'figures', 'res', item)
        for sub_item in os.listdir(item_path):
            if '.txt' in sub_item:
                data = read_soph_txt(os.path.join(item_path, sub_item))
                data = np.array(data)
                
                y = data[:, 1].reshape(1, data.shape[0])
                y = (y-sum(y)/data.shape[0])/(max(y)-min(y)+1e-6)

                peaks = find_peaks_shift(y)
                fit_peaks.append(peaks)
                fit_data.append(data)


    all_avg_dis_x, all_avg_dis_xy = []

    # calculate the mean peaks distance
    for i in range(len(fit_peaks)):
        fit_x = fit_data[i][:, 0]
        org_x = org_data[i][:, 0]
        fit_p = fit_peaks[i]
        org_p = org_peak[i]

        fit_p_new_x = np.array([fit_x[x] - fit_x[fit_p[0]] for x in fit_p])
        fit_p_new_y = np.array([fit_y[x] - fit_y[fit_p[0]] for x in fit_p])
        org_p_new_x = np.array([org_x[x] - org_x[org_p[0]] for x in org_p])
        org_p_new_y = np.array([org_y[x] - org_y[org_p[0]] for x in org_p])

        dev = np.abs(fit_p_new_x - org_p_new_x)
        avg_dis_x = np.mean(dev)

        avg_dis_xy = np.mean(np.sqrt((fit_p_new_x-org_p_new_x)**2+(fit_p_new_y-org_p_new_y)**2)

        all_avg_dis_x.append(avg_dis_x)
        all_avg_dis_xy.append(avg_dis_xy)    


        
    df = pd.DataFrame({'name': names, 'avg_dis_x': all_avg_dis_x, 'avg_dis_x': all_avg_dis_x})
    # df = df.sort_values(by='name', ascending=True)
    # df.to_csv('evaluation.csv', index=False)


    correct_ratio = calculate_ratio_below_threshold(all_avg_dis_x, all_avg_dis_xy, threshold)
    print("correct ratio of spectra is", correct_ratio)