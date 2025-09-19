import os
import numpy as np

path='/zhouzhinuo/data/metal_vacancy/data'
save_path = '/zhouzhinuo/data/metal_vacancy'


files=os.listdir(path)

len = len(files)
count = 0
g1 = np.zeros(len)
g2 = np.zeros(len)
g3 = np.zeros(len)
lwpp = np.zeros(len)

data = np.zeros((len,2000,2))


# g value combination
Gvalue = np.array([1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2.0, 2.02, 2.04, 2.06, 2.08, 2.1, 2.12, 2.14, 2.16])
Gindex = np.array(range(Gvalue.shape[0]))
print(Gvalue.shape[0])

lwppvalue = np.array([1, 2])
lwppindex = np.array(range(lwppvalue.shape[0]))

# read the data
for file in files:
    a_f = file.split('=')

    g1[count] = Gindex[Gvalue==float(a_f[1].split('_')[0])]
    g2[count] = Gindex[Gvalue==float(a_f[2].split('_')[0])]
    g3[count] = Gindex[Gvalue==float(a_f[3].split('_')[0])]
    lwpp[count] = lwppindex[lwppvalue==float(a_f[4].split('.')[0])]
    
    f = open(path+"/"+file);
    iter_f = iter(f)

    c=0
    for line in iter_f:
        data[count][c][0]=float(line.split()[0])
        data[count][c][1]=float(line.split()[1])
        c = c + 1
	
    count = count + 1
    if count%1000 == 0:
        print(count)

    f.close()



# save the data
np.save(os.path.join(save_path, f"data.npy"),data)
np.save(os.path.join(save_path, f"g1.npy"),g1)
np.save(os.path.join(save_path, f"g2.npy"),g2)
np.save(os.path.join(save_path, f"g3.npy"),g3)
np.save(os.path.join(save_path, f"lwpp.npy"),lwpp)

