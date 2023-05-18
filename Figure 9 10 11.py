#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.fft as fft
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.transforms as transforms

import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

get_ipython().run_line_magic('matplotlib', 'inline')
torch.manual_seed(123)


# In[2]:


# dataset loading 
test = pd.read_csv('D:/mnist-digital/mnist_in-csv/mnist_test.csv',dtype=np.float32)

x_test = test.loc[:, test.columns != 'label'].values.reshape(-1,1,28,28) / 255
y_test = test.label.values


# In[3]:


#target: label to Detector 
def Detector_Regions(label):
    # sampling number: 256*256
    detector_regions = torch.zeros(84,84) # detector_regions:(84,84) padding:(58,58,58,58)
    Detector_size = 12 # 4.8mm * 4.8mm   12*7=84
    if label < 3:
        d_h = 0
        d_w = label * ( 3 * Detector_size)
    else: 
        if label < 7:
            d_h = 3 * Detector_size
            d_w = (label-3) * (2 * Detector_size)
        else:
            d_h = 6 * Detector_size
            d_w = (label-7) * ( 3 * Detector_size)
    detector_regions[d_h:d_h+Detector_size,d_w:d_w+Detector_size] = 0.5
    pad = nn.ZeroPad2d(padding=(58,58,58,58))
    detector_regions = pad(detector_regions)
    return detector_regions


# In[4]:


# Energy Distribution
def count_region(output):
    h = output.shape[-2]
    w = output.shape[-1]
    label_class = 10
    count_region = torch.zeros([output.shape[0],label_class])
    for i in range(label_class):
        detector_region = Detector_Regions(i)
        count_region[:,i] = (output.reshape(-1,h,w) * detector_region).sum(axis = [1,2])
    return count_region
        


# In[5]:


# def sampling number
Nx,Ny = [200,200]

to_tensor = transforms.ToTensor()
resize = transforms.Resize([Nx, Ny])

y_test = torch.from_numpy(y_test).type(torch.LongTensor)


# In[6]:


class GetData(Dataset):
    def __init__(self, x_train, y_train, to_tensor, resize):
        self.X = x_train
        self.to_tensor = to_tensor
        self.resize = resize
        self.Y = y_train
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        detector_regions = Detector_Regions(self.Y[index])
        transformed = self.to_tensor(self.X[index]).permute((1, 2, 0)).contiguous()
        transformed = self.resize(transformed)
        #transformed = transformed / torch.sqrt(((transformed)**2).sum() / detector_regions.sum())
        return  transformed,detector_regions.unsqueeze(0)


# In[7]:


testset = GetData(x_test,y_test,to_tensor, resize)
testloader = DataLoader(testset, batch_size=1, shuffle=True) # batch_size=1


# In[8]:


# def diffraction function
class Layer(nn.Module):
    def __init__(self,Lx, Ly, Nx, Ny,lam,k,z,):
        super(Layer,self).__init__()
        self.x = np.linspace(-Lx/2,Lx/2,Nx)
        self.x = self.x.astype(np.float32)   # HTRSD: single  ASM: double
        self.y = np.linspace(-Ly/2,Ly/2,Ny)
        self.y = self.y.astype(np.float32)   # HTRSD: single  ASM: double
        self.xx,self.yy = np.meshgrid(self.x, self.y)
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.r_max = np.max((self.xx**2 + self.yy**2)**0.5)
        self.z_c = Nx * self.dx**2 / lam
        # Angular specturm
        self.fx = np.linspace(-0.5/self.dx,0.5/self.dx,Nx)
        self.fx = self.fx.astype(np.float32)   # HTRSD: single  ASM: double
        self.fy = np.linspace(-0.5/self.dy,0.5/self.dy,Ny)
        self.fy = self.fy.astype(np.float32)   # HTRSD: single  ASM: double
        self.fxx,self.fyy = np.meshgrid(self.fx, self.fy)
        
        #self.phase = nn.Parameter(torch.rand(Nx, Ny)) 
        self.phase = nn.Parameter(torch.zeros(Nx, Ny))
        
    # CTF
    def H(self,z):
        rho = np.abs(np.sqrt(self.fxx**2 + self.fyy**2))
        temp = lam*rho
        cond = temp < 1 
        CTF = np.where(cond, np.exp(1j*k*z*np.sqrt(np.abs(1-temp**2))), np.exp(-k*z*np.sqrt(np.abs(temp**2-1))))
        return torch.from_numpy(CTF)
        
    def H_e(self,z,n):
        rho = np.abs(np.sqrt(self.fxx**2 + self.fyy**2))
        temp = lam*rho
        if n == 1:
            CTF = np.exp(1j*k*z)*np.exp(-1j*k*z*0.5*temp**2)
            return torch.from_numpy(CTF)
        if n == 2:
            CTF = np.exp(1j*k*z)*np.exp(-1j*k*z*0.5*temp**2)*np.exp(-1j*z*k*0.125*temp**4)
            return torch.from_numpy(CTF)
    # HTRSD
    def forward(self, u1, z, n=1):
        U1 = fft.fft2(fft.fftshift(u1))
        if self.dx <= lam:
            CTF = fft.fftshift(self.H(z))
            return fft.fftshift(fft.ifft2(U1 * CTF))
        if z <= self.z_c:
            CTF = fft.fftshift(self.H_e(z,n))
            return fft.fftshift(fft.ifft2(U1 * CTF))



# In[9]:


# def model:  Multilayer  K = 5
class DiffNet(nn.Module):
    def __init__(self):
        super(DiffNet,self).__init__()
        self.input = Layer(Lx,Ly,Nx,Ny,lam,k,d)  # no phase 
        self.difflayer1 = Layer(Lx,Ly,Nx,Ny,lam,k,d)
        self.difflayer2 = Layer(Lx,Ly,Nx,Ny,lam,k,d)
        self.difflayer3 = Layer(Lx,Ly,Nx,Ny,lam,k,d)
        self.difflayer4 = Layer(Lx,Ly,Nx,Ny,lam,k,d)
        self.difflayer5 = Layer(Lx,Ly,Nx,Ny,lam,k,d)
        self.fc1 = nn.Sigmoid()
    def forward(self,u0):
        out = self.input(u0, d)
        out = out * torch.exp(1j*2*torch.pi*(self.fc1(self.difflayer1.phase)-0.5))
        out = self.difflayer1(out, d)
        out = out * torch.exp(1j*2*torch.pi*(self.fc1(self.difflayer2.phase)-0.5))
        out = self.difflayer2(out, d)
        out = out * torch.exp(1j*2*torch.pi*(self.fc1(self.difflayer3.phase)-0.5))
        out = self.difflayer3(out, d)
        out = out * torch.exp(1j*2*torch.pi*(self.fc1(self.difflayer4.phase)-0.5))
        out = self.difflayer4(out, d)
        out = out * torch.exp(1j*2*torch.pi*(self.fc1(self.difflayer5.phase)-0.5))
        out = self.difflayer5(out, d)
        out = torch.abs(out)**2
        out = (out / (out.sum(axis=[-1,-2]).unsqueeze(-1) * torch.ones([1,200,200]).cuda()).unsqueeze(1) * 144*3)
        return out
    
# 此模型的输出是float,注意数据类型匹配 Double but expect Float

# def D2NN parameters
mm = 1e-3
lam = 0.75 * mm
k = 2 * np.pi / lam # wave number
# propagation distance
d = 30 * mm
Lx = 80 * mm
Ly = 80 * mm
dx = Lx / Nx


# In[10]:


D2NN = torch.load('./linear_5.pt')
D2NN


# In[11]:


m = nn.Sigmoid()
phase_mask = []
for name, param in D2NN.named_parameters():
    print(name)
    phase = 2*torch.pi*(m(param.detach().cpu())-0.5)
    print(phase)
    phase_mask.append(phase)
    print("requires_grad:", param.requires_grad)
    print("-----------------------------------")
#phase_mask = torch.cat(phase_mask,dim=0)


# In[41]:


fig, ax = plt.subplots()
ax.imshow(phase_mask[1], cmap = plt.get_cmap('RdYlBu'))

norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
cmap = mpl.cm.RdYlBu
cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap))
cb.ax.set_title('Phase')
cb.set_ticks(ticks=[0,2*np.pi],labels=['0','$2\pi$'])
\
ax.axis('off')

fig.show()


# In[ ]:


fig, ax = plt.subplots()
ax.imshow(phase_mask[2], cmap = plt.get_cmap('RdYlBu'))

ax.axis('off')

fig.show()


# In[ ]:


fig, ax = plt.subplots()
ax.imshow(phase_mask[3], cmap = plt.get_cmap('RdYlBu'))

ax.axis('off')

fig.show()


# In[ ]:


fig, ax = plt.subplots()
ax.imshow(phase_mask[4], cmap = plt.get_cmap('RdYlBu'))

ax.axis('off')

fig.show()


# In[ ]:


fig, ax = plt.subplots()
ax.imshow(phase_mask[5], cmap = plt.get_cmap('RdYlBu'))

ax.axis('off')

fig.show()


# In[ ]:


input_digit=testset[0]

digit_image = input_digit[0]
ground_truth = input_digit[1]


# In[ ]:


diffraction = Layer(Lx,Ly,Nx,Ny,lam,k,d)


# In[ ]:


Amplitude =[]
Amplitude.append(digit_image)

Phase=[]
Phase.append(diffraction.phase.detach())


# In[ ]:


inputs=digit_image
for i in range(5):
    out = diffraction(inputs,d) * torch.exp(1j*phase_mask[i+1])
    
    Amplitude.append(torch.abs(out))
    Phase.append(torch.angle(out))
    
    inputs = out
    
out = diffraction(inputs,d)
Amplitude.append(torch.abs(out))
Phase.append(torch.angle(out))

out = torch.abs(out)**2


# In[ ]:


fig, axs = plt.subplots(1,2)
axs[0].imshow(255-Amplitude[0].reshape(200,200),cmap=plt.cm.binary)
axs[1].imshow(Phase[0].reshape(200,200),cmap='RdYlBu_r')

axs[0].axis('off')
axs[1].axis('off')

fig.show()


# In[ ]:


fig, axs = plt.subplots(1,3)
axs[0].imshow(255-Amplitude[1].reshape(200,200),cmap=plt.cm.binary)
axs[1].imshow(Phase[1].reshape(200,200),cmap='RdYlBu_r')
axs[2].imshow(phase_mask[1], cmap = plt.get_cmap('RdYlBu'))
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
fig.show()


# In[ ]:


fig, axs = plt.subplots(1,3)
axs[0].imshow(255-Amplitude[2].reshape(200,200),cmap=plt.cm.binary)
axs[1].imshow(Phase[2].reshape(200,200),cmap='RdYlBu_r')
axs[2].imshow(phase_mask[2], cmap = plt.get_cmap('RdYlBu'))
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
fig.show()


# In[ ]:


fig, axs = plt.subplots(1,3)
axs[0].imshow(255-Amplitude[3].reshape(200,200),cmap=plt.cm.binary)
axs[1].imshow(Phase[3].reshape(200,200),cmap='RdYlBu_r')
axs[2].imshow(phase_mask[3], cmap = plt.get_cmap('RdYlBu'))
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
fig.show()


# In[ ]:


fig, axs = plt.subplots(1,3)
axs[0].imshow(255-Amplitude[4].reshape(200,200),cmap=plt.cm.binary)
axs[1].imshow(Phase[4].reshape(200,200),cmap='RdYlBu_r')
axs[2].imshow(phase_mask[4], cmap = plt.get_cmap('RdYlBu'))
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
fig.show()


# In[ ]:


fig, axs = plt.subplots(1,3)
axs[0].imshow(255-Amplitude[5].reshape(200,200),cmap=plt.cm.binary)
axs[1].imshow(Phase[5].reshape(200,200),cmap='RdYlBu_r')
axs[2].imshow(phase_mask[5], cmap = plt.get_cmap('RdYlBu'))
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
fig.show()


# In[ ]:


fig, axs = plt.subplots(1,2)
axs[0].imshow(255-Amplitude[6].reshape(200,200),cmap=plt.cm.binary)
axs[1].imshow(Phase[6].reshape(200,200),cmap='RdYlBu_r')

axs[0].axis('off')
axs[1].axis('off')

fig.show()


# In[ ]:


fig, axs = plt.subplots(2,2)
axs[0,0].imshow(255-digit_image.reshape(200,200),cmap=plt.cm.binary)
axs[0,1].imshow(255-ground_truth.reshape(200,200),cmap=plt.cm.binary)
axs[1,0].imshow(255-out.detach().reshape(200,200),cmap=plt.cm.binary)
axs[0,0].axis('off')
axs[0,1].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
fig.show()


# In[ ]:


detector=torch.zeros([200,200])
for i in range(10):
    detector += Detector_Regions(i)


# In[ ]:


fig, axs = plt.subplots(2,2)
axs[0,0].imshow(255-digit_image.reshape(200,200),cmap=plt.cm.binary)
axs[0,1].imshow(255-detector.reshape(200,200),cmap=plt.cm.binary)
axs[1,0].imshow(255-out.detach().reshape(200,200),cmap=plt.cm.binary)
axs[0,0].axis('off')
axs[0,1].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
fig.show()


# In[ ]:




