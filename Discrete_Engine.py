import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

### DATA MANIPULATION ###
# access the base dataset
dataset = pd.read_csv('./Mel.csv')

# some data manipulation:
# extract the data out of the CSV file
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values

# normalizing the data (currently not in use)
#X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y)

# divide the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

### NET CONSTRUCTION ####
class DeepNet(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()

		# Layer one, so on so forth (linear layer)
		self.layer1 = nn.Linear(input_dim, 64)
		self.layer2 = nn.Linear(64, 32)
		self.layerOutput = nn.Linear(32, output_dim)
	def forward(self, x):

		# installing the activation function (RELU function)
		y_pred1 = F.relu(self.layer1(x))
		y_pred2 = F.relu(self.layer2(y_pred1))

		return self.layerOutput(y_pred2)

# standard procedure: determine the architecture, SGD method, and the loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
regressor = DeepNet(8, 1).to(device)
optimizer = optim.Adam(regressor.parameters(), lr=0.0005)
criterion = torch.nn.MSELoss()

# Training and Testing data subdivision and control.
ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True)
ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=True)

### TRAINING PHASE ###
EPOCHS = 50
for epoch in range(EPOCHS):
	for data in dataloader_train:

		# Find the value that the net will produce with existing weight and bias configuration
		y_pred_output = regressor(X_train)

		# calculate the loss between the target and estimated output
		loss = criterion(y_pred_output, y_train)

		# commit SGD on the known net
		optimizer.zero_grad()

		# propagate the optimization backwards
		loss.backward()
		optimizer.step()

	# print out the losses for each iteration
	print(loss)

### TESTING PHASE ###
# deactivates the net's ability to be trained: enter testing phase
regressor.eval()
with torch.no_grad():
	
	# Same principles apply here
    test_out = regressor(X_test)
    loss = criterion(test_out, y_test)
    print("testing loss:", loss)

### PERFORMANCE REVIEW ###
# in theory, an accurate model will yield an estimation that is same as the 
# actual target value in the dataset, thus we test a known point
X_response_test = [140, 4.2, 215.9, 194, 4.7, 1050, 710, 28]
X_response_test = torch.tensor(X_response_test).float()
test_response = regressor(X_response_test)
print("test response:", test_response)
