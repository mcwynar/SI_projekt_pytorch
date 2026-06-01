"""
Created on Tue Nov 29 20:16:02 2022

@author: rz
based on
https://janakiev.com/blog/pytorch-iris/
Missing experiment on lr
"""

import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cm = 1/2.54  # centimeters in inches

x,y_t,x_norm,x_n_s,y_t_s = hkl.load('zoo.hkl')
# x,y_t,x_norm = hkl.load('wine.hkl')
# x,y_t,x_norm,x_n_s,y_t_s = hkl.load('haberman.hkl')
# x,y_t,x_norm,x_n_s,y_t_s = hkl.load('iris.hkl')

if min(y_t.T)[0] > 0:
    y=y_t.squeeze()-1 #index of first class should equal to 0
else:
    y=y_t.squeeze()

X=x.T

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)


# Configure Neural Network Models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, K1, K2):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, K1)
        self.layer2 = nn.Linear(K1, K2)
        self.layer3 = nn.Linear(K2, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 ])
K1_vec = np.arange(1,11,1)
K2_vec = K1_vec
PK_2D_K1K2 = np.zeros([len(K1_vec),len(K2_vec)])
max_epoch  = 100
PK_2D_K1K2_max = 0
k1_ind_max = 0
k2_ind_max = 0

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        model     = Model(X_train.shape[1], int(max(y)+1), K1_vec[k1_ind], K2_vec[k2_ind])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_vec[0])
        loss_fn   = nn.CrossEntropyLoss()
        # print(model)

        for epoch in range(max_epoch):
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)

            # Zero gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            PK = correct.mean().item()*100
            print("K1 {} | K2 {} | PK {}". format(K1_vec[k1_ind], K2_vec[k2_ind], PK))
            PK_2D_K1K2[k1_ind, k2_ind] = PK

        if PK > PK_2D_K1K2_max:
            PK_2D_K1K2_max = PK
            k1_ind_max = k1_ind
            k2_ind_max = k2_ind


print("OPTYMALNE WARTOŚCI K1K2: K1={} | K2={} | lr={} | PK={}".\
      format(K1_vec[k1_ind_max], K2_vec[k2_ind_max], lr_vec[0], \
             PK_2D_K1K2[k1_ind_max, k2_ind_max]))
fig = plt.figure(figsize=(20*cm, 20*cm))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(X, Y, PK_2D_K1K2.T, cmap='viridis')

ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.savefig("Fig.1_PK_K1K2_pytorch_iris.png",bbox_inches='tight')