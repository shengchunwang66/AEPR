

## Data preparation
'''
python generate_mixture.py  # Generate a value for mixture
'''

'''
python generate_single.py  # Generate a value for spin-trapping
'''

After generating the parameter combinations, we use easyspin to generate the simulated file.



## Training

During the train of EPRNet, we first combine all the simulated file together with different g value.
'''
pyton read.py
'''

Then we preprocess and split the data. 
'''
python preprocess.py
'''

Then we transform the data to GramianAngularField
'''
python 1dto2d.py
'''

Finally, we train the network.

'''
python train.py
'''



## Evaluation

This code is for evaluating the results of EPRNet.

```
python evaluation.py
```

