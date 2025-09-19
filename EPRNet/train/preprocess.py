import numpy as np
import matplotlib.pyplot as plt
import os
import numpy
from sklearn.preprocessing import MinMaxScaler


save_path = '/home/ubuntu/data/metal_vacancy/'



data1 = np.load('/home/ubuntu/data/metal_vacancy/data.npy')
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
tmp = scaler.fit_transform(np.transpose(data1[:,:,1]))
data1[:,:,1] = np.transpose(tmp)
data1 = data1.astype('float32')
print(data1.dtype)

ind = np.arange(2000)


a_max = np.max(ind[np.abs(data1[0,:,1])>0.1])
a_min = np.min(ind[np.abs(data1[0,:,1])>0.1])
split_l = int(np.floor(0.2 * a_min))
split_r = int(np.floor(0.2 * (1024-a_max)))

dataset_size = len(data1[:,0,0])
split = int(np.floor(0.9995 * dataset_size))

p = np.random.permutation(len(data1))

data_shuffle = data1[p,:,:]

data_train = data_shuffle


data_final = np.zeros([data_train.shape[0], 2000, 2])
print('iterion start!\n\n')


for i in range(data_train.shape[0]):

    a = np.expand_dims(data_train[i, :, 0], axis=1)
    aa = np.transpose(a)
    a = None
    data_final[1*i:(1*i+1), :, 0] = aa
    aa = None

    a = np.expand_dims(data_train[i, :, 1], axis=1)
    aa = np.transpose(a)
    a = None
    data_final[1*i:(1*i+1), :, 1] = aa
    aa = None

    if i%1000==0:
        print(i)


print('iterion end!\n\n')

noise_final = np.random.normal(0, 0.01, data_final[:,:,1].shape)
print('noise add!\n\n')

for i in range(data_final.shape[1]):
    noise = np.random.normal(0, 0.01, data_final.shape[0])
    data_final[:,i,1] = data_final[:,i,1] + 0.1*noise
    if i%100==0:
        print(i)
        print(noise)

np.save(os.path.join(save_path,'train_jinshu.npy'), data_final)


print('save train_jinshu.npy and val_jinshu.npy succeed! \n\n')

fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
plt.plot(data_final[3,:,1])
plt.axis('off')
plt.savefig(f'images.png', bbox_inches='tight', pad_inches=0)
plt.clf()


dataset_size = len(p)
# # indices = list(range(dataset_size))
split = int(np.floor(0.9995 * dataset_size))
print('split', split)

g1 = np.load(save_path+'g1.npy')
g1_shuffle = g1[p]
g1_aug_train = np.repeat(g1_shuffle, 1, axis=0)

g2 = np.load(save_path+'g2.npy')
g2_shuffle = g2[p]
g2_aug_train = np.repeat(g2_shuffle, 1, axis=0)

g3 = np.load(save_path+'g3.npy')
g3_shuffle = g3[p]
g3_aug_train = np.repeat(g3_shuffle, 1, axis=0)

lwpp = np.load(save_path+'lwpp.npy')
lwpp_shuffle = lwpp[p]
lwpp_aug_train = np.repeat(lwpp_shuffle, 1, axis=0)


np.save(save_path+'g1_train.npy', g1_aug_train)
np.save(save_path+'g2_train.npy', g2_aug_train)
np.save(save_path+'g3_train.npy', g3_aug_train)
np.save(save_path+'lwpp_train.npy', lwpp_aug_train)

# print(data_test[0,0:10,1])
print('save aug npy succeed! \n\n')

