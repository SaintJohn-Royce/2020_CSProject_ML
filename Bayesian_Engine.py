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

# git add Bayesian_Engine.py
# git commit -m ""
# git remote add origin https://github.com/SaintJohn-Royce/2020_CSProject_ML.git
# git push -u origin master

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
@variational_estimator
class DeepNet(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		# Layer one, so on so forth (Bayesian Layer)
		self.blinear1 = BayesianLinear(input_dim, 128)
		self.blinear2 = BayesianLinear(128, 32)
		self.blinearOutput = BayesianLinear(32, output_dim)
		
	def forward(self, x):

		# installing the activation function (RELU function)
		x_1 = F.relu(self.blinear1(x))
		x_2 = F.relu(self.blinear2(x_1))

		return self.blinearOutput(x_2)

# standard procedure: determine the architecture, SGD method, and the loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
regressor = DeepNet(8, 1).to(device)
optimizer = optim.Adam(regressor.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training and Testing data subdivision and control.
ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True)
ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=True)

### TRAINING PHASE ###
EPOCHS = 30
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

	# print out the losses for each iteration (currently not in use)
	#print(loss)

### TESTING PHASE ###
# deactivates the net's ability to be trained: enter testing phase
regressor.eval()
with torch.no_grad():

	# Same principles apply here
    test_out = regressor(X_test)
    loss = criterion(test_out, y_test)
    print("========================================================")
    print("testing loss:   ", loss)
    print("========================================================")

### PERFORMANCE REVIEW ###
# in theory, an accurate model will yield an estimation that is same as the 
# actual target value in the dataset, thus we test a known point
while True:

	# user input of data points
	cement          = input("cement                 (kg/m^3) [100-600]: 	")
	slag            = input("slag                   (kg/m^3) [000-400]: 	")
	ash             = input("ash                    (kg/m^3) [000-350]: 	")
	water           = input("water                  (kg/m^3) [100-300]: 	")
	sup_plasticizer = input("super-plasticizer      (kg/m^3) [000-050]: 	")
	c_Aggre         = input("Coarse-aggregate       (kg/m^3) [500-1300]: 	")
	f_Aggre         = input("Fine-aggregate         (kg/m^3) [500-1300]: 	")
	age             = input("age of sample          (day(s)) [001-365]: 	")

	X_response_test = [float(cement), float(slag), float(ash), float(water), 
					   float(sup_plasticizer), float(c_Aggre), float(f_Aggre), float(age)]
	X_response_test = torch.tensor(X_response_test).float()

	# tune parameters
	samples = 100
	std_multiplier = 2

	# By running iterations, we can calculate the mean and standard deviation of
	# the Gaussian distribution of the posterior
	def regression_test(regressor, X_response_test, samples, std_multiplier):

		# create a list of [regressor(X), ... ... regressor(X)]
		prediction = [regressor(X_response_test) for i in range(samples)]
		prediction = torch.stack(prediction)

		# standard Mean, Standard Deviation, and Confidence Interval calculation
		means = prediction.mean(axis=0)
		stds = prediction.std(axis=0)
		ci_upper = means + (std_multiplier * stds)
		ci_lower = means - (std_multiplier * stds)
		print("mean: 		", means)
		print("STD: 		", stds)
		print("upper CL: 	", ci_upper)
		print("lower CL: 	", ci_lower)
		return

	# actually running the regression test
	regression_test(regressor, X_response_test, samples, std_multiplier)

	test_continuity = input("conduct another test? [Y]/[N]: ")

	print("========================================================")

	if test_continuity == "N":

		break

