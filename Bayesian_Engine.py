import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy
import pandas as pd

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./Mel.csv')


X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values
#X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

@variational_estimator
class DeepNet(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.blinear1 = BayesianLinear(input_dim, 128)
		self.blinear2 = BayesianLinear(128, 32)
		self.blinearOutput = BayesianLinear(32, output_dim)
	def forward(self, x):
		x_1 = F.relu(self.blinear1(x))
		x_2 = F.relu(self.blinear2(x_1))

		return self.blinearOutput(x_2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
regressor = DeepNet(8, 1).to(device)
optimizer = optim.Adam(regressor.parameters(), lr=0.0005)
criterion = torch.nn.MSELoss()

ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True)

ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=True)

EPOCHS = 30
for epoch in range(EPOCHS):
	for data in dataloader_train: 
		y_pred_output = regressor(X_train)
		loss = criterion(y_pred_output, y_train)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	#print(loss)

regressor.eval()
with torch.no_grad():
    test_out = regressor(X_test)
    loss = criterion(test_out, y_test)
    print("testing loss:   ", loss)

X_response_test = [475, 0, 0, 228, 0, 932, 594, 365]
X_response_test = torch.tensor(X_response_test).float()
std_multiplier = 2
samples = 200

def regression_test(regressor, X_response_test, samples):
	prediction = [regressor(X_response_test) for i in range(samples)]
	prediction = torch.stack(prediction)
	means = prediction.mean(axis=0)
	stds = prediction.std(axis=0)
	ci_upper = means + (std_multiplier * stds)
	ci_lower = means - (std_multiplier * stds)
	print("mean: 		", means)
	print("STD: 		", stds)
	print("upper CL: 	", ci_upper)
	print("lower CL: 	", ci_lower)
	return

regression_test(regressor, X_response_test, samples)

