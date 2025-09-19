import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import numpy as np
from pyts.image import MarkovTransitionField
import os



#path='/mnt/workspace/zhangshufei_01/zhang/jinshu_train_data'
# path='/mnt/workspace/zhangshufei_01/zhang/jinshu_test_data'
save_path = '/home/ubuntu/data/metal_vacancy/'
save_path_train = '/home/ubuntu/data/metal_vacancy/train/'

data_aug_train = np.load(save_path+'train_jinshu.npy')
print(data_aug_train.shape)  # (575153, 7000, 2)


# train_size = len(data_aug_train)

train_size = 1
# test_size = 288
data_aug_train2d_process = np.zeros((train_size,3,224,224))


# print(data_aug_train2d_process.shape)

print('aaaaaaaa')

# print(np.max(data_aug_train2d_process[0,0,:,:]))
# print(np.min(data_aug_train2d_process[0,0,:,:]))

gasf = GramianAngularField(image_size=224, method = 'summation')
gadf = GramianAngularField(image_size=224, method = 'difference')
#mtf = MarkovTransitionField(image_size=224, n_bins=8, strategy='quantile')



# for i in range(515207, len(data_aug_train)):
# for i in range(515206, 515207):
for i in range(len(data_aug_train)):
    # if os.path.exists(save_path_train+'/jinshu_train_data'+str(i)+'.npy'):
    #    continue 
    data_gasf_train = gasf.fit_transform(data_aug_train[i*train_size:(i+1)*train_size,:,1])
    data_gadf_train = gadf.fit_transform(data_aug_train[i*train_size:(i+1)*train_size,:,1])
    #data_mtf_train = mtf.fit_transform(data_aug_train[i*train_size:(i+1)*train_size,:,1])

    #data_gasf_test = gasf.fit_transform(data_aug_val[0:test_size,:,1])
    #data_gadf_test = gadf.fit_transform(data_aug_val[0:test_size,:,1])
    #data_mtf_test = mtf.fit_transform(data_aug_val[0:test_size,:,1])

    #data_aug_train2d_process[:,0,:,:] = data_mtf_train[0:]
    data_aug_train2d_process[:,1,:,:] = data_gasf_train[0:]
    data_aug_train2d_process[:,2,:,:] = data_gadf_train[0:]
    data_aug_train2d_process[:,0,:,:] = data_aug_train2d_process[:,1,:,:]

    #data_aug_val2d_process[:,0,:,:] = data_mtf_test[0:]
    #data_aug_val2d_process[:,1,:,:] = data_gasf_test[0:]
    #data_aug_val2d_process[:,2,:,:] = data_gadf_test[0:]
    print(save_path_train+'/jinshu_train_data'+str(i)+'.npy')
    np.save(save_path_train+'/jinshu_train_data'+str(i)+'.npy', data_aug_train2d_process) 

    if i % 100==0:
        print(i)
    #np.save('spin_val_GramianAngularField2.npy', data_aug_val2d_process) 




#data_mtf_test = mtf.fit_transform(data_aug_val[0:test_size,:,1])




