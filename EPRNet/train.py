# 导入必要的库
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch import distributed as dist
from torch.utils.data import TensorDataset,SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


os.environ["RANK"] = '0'
os.environ["WORLD_SIZE"] = '1'
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '12356'
torch.distributed.init_process_group(backend="nccl")


class MyDataset(Dataset):
    def __init__(self, data_prefix, label_file, label_file1, label_file2, label_file3):
        self.data_prefix = data_prefix # prefix of the data file
        self.labels = label_file # load the label file
        self.labels1 = label_file1
        self.labels2 = label_file2
        self.labels3 = label_file3

    def __len__(self):
        return len(self.labels) # return the length of the dataset
    
    def __getitem__(self, index):
        # load the npy file according to the index
        data_file = '/home/ubuntu/data/metal_vacancy/train/' + self.data_prefix + str(index) + '.npy'
        
        data = np.load(data_file)
        if data.shape == 0:
            print(data_file)
            exit(0)
        # return the data and the label
        return torch.tensor(data), torch.tensor(self.labels[index]), torch.tensor(self.labels1[index]), torch.tensor(self.labels2[index]), torch.tensor(self.labels3[index])


def compare_and_swap(a, b):
    # check if the length of the two vectors is equal, if not, return the error message
    if len(a) != len(b):
        return "Error: the two vectors must have the same length."
    # use the list comprehension to generate a new vector c, the element is the larger one of a and b
    c = [max(a[i], b[i]) for i in range(len(a))]
    # use the list comprehension to generate a new vector d, the element is the smaller one of a and b
    d = [min(a[i], b[i]) for i in range(len(a))]
    # return the two vectors after the swap
    return c, d

def compare_and_swap_multi(a, b, c, d):
    # check if the length of the two vectors is equal, if not, return the error message
    if len(a) != len(b):
        return "Error: the two vectors must have the same length."
    if len(c) != len(b):
        return "Error: the two vectors must have the same length."
    if len(c) != len(d):
        return "Error: the two vectors must have the same length."

    a = np.expand_dims(a,axis=1)
    b = np.expand_dims(b,axis=1)
    c = np.expand_dims(c,axis=1)
    d = np.expand_dims(d,axis=1)
 
    e = np.concatenate((a,b,c,d),axis=1)
    e = np.sort(e, axis=1)
    a0 = e[:,3]
    a1 = e[:,2]
    a2 = e[:,1]
    a3 = e[:,0]
    return a0, a1, a2, a3

class MyNet(nn.Module):
    def __init__(self, input_dim, model):
        super(MyNet, self).__init__()
        # define the three parallel three-layer neural networks
        # each network has two hidden layers with 256 units and ReLU activation
        # the output layer has the same number of units as the number of classes
        self.model = model
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 16)  # nn.Linear(200, 7)
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 16)  # nn.Linear(200, 7)
        )
        self.net3 = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 16)  # nn.Linear(200, 7)
        )
        self.net4 = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )
        

    def forward(self, x):
        x = self.model(x) # Get the features from ResNet50
        out1 = self.net1(x) # Get the output from net1
        out2 = self.net2(x) # Get the output from net2
        out3 = self.net3(x) # Get the output from net3
        out4 = self.net4(x)
        return out1, out2, out3, out4
# define the hyper-parameters
#batch_size = 32 # the number of images in each batch
num_epochs = 500 # the number of epochs
learning_rate = 0.00001 # the learning rate

# load the dataset, here we use the randomly generated data as an example
# you can replace it with your own data
#train_data = torch.rand(1000, 3, 224, 224) # randomly generate 1000 3-channel 224*224 images
#train_labels = torch.rand(1000, 3) # randomly generate 1000 3-dimensional vectors as labels
def preprocess(images):
    # create a transform object, including the following operations:
    # 1. convert the image to PIL format
    # 2. randomly flip the image horizontally
    # 3. randomly crop the image, output size is [224,224]
    # 4. convert the image to tensor format
    # 5. normalize the image, using the mean and standard deviation of ImageNet
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    output = torch.empty(images.shape[0], 3, 224, 224)
    # create a empty tensor, for storing the preprocessed images, size is [1000,3,224,224]
    # output = torch.empty(1000, 3, 224, 224)
    
    # preprocess each image, and store the result in the output tensor
    for i in range(images.shape[0]):
        output[i] = transform(images[i])
    
    # return the output tensor
    return output

my_path = '/home/ubuntu/data/metal_vacancy'


g1_aug_train = np.load(my_path+"/"+'g1_train.npy')
g2_aug_train = np.load(my_path+"/"+'g2_train.npy')
g3_aug_train = np.load(my_path+"/"+'g3_train.npy')
lwpp_aug_train = np.load(my_path+"/"+'lwpp_train.npy')


g1_aug_train = np.expand_dims((g1_aug_train-np.mean(0)), axis=1)
g2_aug_train = np.expand_dims((g2_aug_train-np.mean(0)), axis=1)
g3_aug_train = np.expand_dims((g3_aug_train-np.mean(0)), axis=1)
lwpp_aug_train = np.expand_dims((lwpp_aug_train-np.mean(0)), axis=1)


train_labels1 = np.concatenate((g1_aug_train, g2_aug_train, g3_aug_train, lwpp_aug_train), axis=1)
trainsize = len(g1_aug_train)

train_labels = torch.tensor(train_labels1[0:trainsize]).float()

target1 = train_labels[:,0].unsqueeze(-1) # [N, 1]
target2 = train_labels[:,1].unsqueeze(-1) # [N, 1]
target3 = train_labels[:,2].unsqueeze(-1) # [N, 1]
target4 = train_labels[:,3].unsqueeze(-1) # [N, 1]


i1 = torch.arange(16) # [M]
i2 = torch.arange(16) # [M]
i3 = torch.arange(16) # [M]
i4 = torch.arange(2) # [M]
sigma = 1.0
scale = 10
train_l_1 = scale* torch.exp(-0.5 * ((i1 - target1) / (sigma*0.5)) ** 2) / (sigma*0.5 * 2.5) # [N, M]
train_l_2 = scale* torch.exp(-0.5 * ((i2 - target2) / (sigma*0.5)) ** 2) / (sigma*0.5 * 2.5) # [N, M]
train_l_3 = scale* torch.exp(-0.5 * ((i3 - target3) / (sigma*0.5)) ** 2) / (sigma*0.5 * 2.5) # [N, M]
train_l_4 = scale* torch.exp(-0.5 * ((i4 - target4) / sigma) ** 2) / (sigma*0.5 * 2.5) # [N, M]



softmax = nn.Softmax(dim=1)

print((train_l_1)[0],train_labels1[0,0])
print((train_l_2)[0],train_labels1[0,1])
print((train_l_3)[0],train_labels1[0,2])

batch_size = 32  # 128
shuffle_dataset = True
random_seed= 42



# load the model, here we use resnet18 as an example
model = torchvision.models.resnet50(pretrained=True) # load the pre-trained resnet18 model

# modify the last layer of the model, so that it outputs a 3-dimensional vector


fc_inputs = model.fc.in_features
model.fc = nn.Identity()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# define three parallel linear layers, for different classification tasks
mynet = MyNet(input_dim=fc_inputs, model=model)


for name, param in mynet.named_parameters():
    if name.startswith("layer1") or name.startswith("layer2"):
        param.requires_grad = True

# move the model to GPU, if there is a available GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mynet.to(device)

local_rank = torch.distributed.get_rank()
mynet = mynet.cuda(local_rank)
torch.cuda.set_device(local_rank)
mynet = torch.nn.parallel.DistributedDataParallel(mynet,
                                                  output_device=local_rank,
                                                  find_unused_parameters=False,
                                                  broadcast_buffers=False)



train_dataset = MyDataset(data_prefix='jinshu_train_data', label_file=train_l_1, label_file1=train_l_2, label_file2=train_l_3, label_file3=train_l_4)
# create the data loader object
sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

indices = list(range(trainsize))  # 369275
np.random.seed(42)
np.random.shuffle(indices)
train_indices = indices
train_sampler = SubsetRandomSampler(train_indices)

train_loader = torch.utils.data.DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=64,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)

# define the loss function and optimizer
#criterion = nn.MSELoss() # use the mean squared error as the loss function
#criterion = nn.L1Loss()
criterion = nn.CrossEntropyLoss()
#torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(mynet.parameters(), lr=learning_rate,weight_decay=1e-6) # use the stochastic gradient descent as the optimizer
#scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
# train the model

train_ANacc = np.array([0.0])
train_AHacc = np.array([0.0])
train_A0P5acc = np.array([0.0])


best_loss = 1000.0 # initialize the best accuracy to 0.0
best_acc = 0.0
best_model_path = 'best_vacancy_resnet.pth' # define the path to save the best model

for epoch in range(num_epochs): # iterate over each epoch
    train_loss = 0.0 # record the cumulative loss of the current epoch
    t_len = 0
    train_acc_g1 = 0.0
    train_acc_g2 = 0.0
    train_acc_g3 = 0.0
    train_acc_lwpp = 0.0
    train_acc_sum = 0.0
    sampler.set_epoch(epoch)
    with tqdm(train_loader, unit="batch") as train_loader:
        for i, (inputs, labels1, labels2, labels3, labels4) in enumerate(train_loader): # iterate over each batch
            # move the input and label to GPU, if there is a available GPU
            inputs = inputs[:,0,:,:,:].float().to(device)
            labels1 = labels1.float().to(device)
            labels2 = labels2.float().to(device)
            labels3 = labels3.float().to(device)  
            labels4 = labels4.float().to(device)

            # clear the gradient cache
            optimizer.zero_grad()
            
            # forward propagation, calculate the output and loss
            outputs1, outputs2, outputs3, outputs4 = mynet(inputs)

            loss1 = criterion(outputs1, labels1)
            loss2 = criterion(outputs2, labels2)
            loss3 = criterion(outputs3, labels3)
            loss4 = criterion(outputs4, labels4)


            loss = loss1 + loss2 + loss3 + loss4
            # backward propagation, update the parameters
            loss.backward()
            optimizer.step()
            # scheduler.step()

            
            train_loss += loss.item()

            ind1 = torch.max(outputs1, dim=1)[1]
            ind2 = torch.max(outputs2, dim=1)[1]
            ind3 = torch.max(outputs3, dim=1)[1]
            ind4 = torch.max(outputs4, dim=1)[1]

            ind1l = torch.max(labels1, dim=1)[1]
            ind2l = torch.max(labels2, dim=1)[1]
            ind3l = torch.max(labels3, dim=1)[1]
            ind4l = torch.max(labels4, dim=1)[1]

            train_acc_g1 += (torch.abs(ind1 - ind1l)<=0.01).sum().item()
            train_acc_g2 += (torch.abs(ind2 - ind2l)<=0.01).sum().item()
            train_acc_g3 += (torch.abs(ind3 - ind3l)<=0.01).sum().item()
            train_acc_lwpp += (torch.abs(ind4 - ind4l)<=0).sum().item()

            train_acc_sum += ((torch.abs(ind1 - ind1l)<=0.01) & (torch.abs(ind2 - ind2l)<=0.01) & (torch.abs(ind3 - ind3l)<=0.01)).sum().item()
            mynet.eval() # set the model to evaluation mode

            t_len += labels1.size()[0]

       
 
        if train_acc_sum > best_acc: # if the current accuracy is better than the best accuracy
            best_loss =train_acc_sum # update the best accuracy
            torch.save(mynet.state_dict(), best_model_path) # save the current model as the best model
            print('Best model saved at epoch %d' % (epoch + 1)) # print the information of the best model
        print('train_acc_sum', train_acc_sum/t_len)

    
print('Finished Training')
np.save('train_ANacc_curve.npy',train_ANacc)
np.save('train_AHacc_curve.npy',train_AHacc)
np.save('train_A0P5acc_curve.npy',train_A0P5acc)

