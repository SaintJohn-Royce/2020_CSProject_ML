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

import wandb

# git add Bayesian_Engine.py
# git commit -m ""
# git remote add origin https://github.com/SaintJohn-Royce/2020_CSProject_ML.git
# git push -u origin master

# procedure and module control
wandB_control = False
validTest_control = False 		
mast_Test_control = False
indivTest_control = True


# WandB documentation
if wandB_control:
	wandb.init(project="Bayesian_Engine.py")


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


### HYPERPARAMETER LISTING ###
hyperparameter_defaults = dict(
	blayer1 = 128,
	blayer2 = 32,
	batch_size = 1,
	learning_rate = 0.001,
	EPOCHS = 31,
	samples = 200)

blayer1 = hyperparameter_defaults["blayer1"]
blayer2 = hyperparameter_defaults["blayer2"]
batch_size = hyperparameter_defaults["batch_size"]
learning_rate = hyperparameter_defaults["learning_rate"]
EPOCHS = hyperparameter_defaults["EPOCHS"]
samples = hyperparameter_defaults["samples"]

### NET CONSTRUCTION ####
@variational_estimator
class DeepNet(nn.Module):
	def __init__(self, input_dim, output_dim, blayer1, blayer2):
		super().__init__()
		# Layer one, so on so forth (Bayesian Layer)
		self.blinear1 = BayesianLinear(input_dim, blayer1)
		self.blinear2 = BayesianLinear(blayer1, blayer2)
		self.blinearOutput = BayesianLinear(blayer2, output_dim)
		
	def forward(self, x):

		# installing the activation function (RELU function)
		x_1 = F.relu(self.blinear1(x))
		x_2 = F.relu(self.blinear2(x_1))

		return self.blinearOutput(x_2)


# standard procedure: determine the architecture, SGD method, and the loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
regressor = DeepNet(8, 1, blayer1, blayer2).to(device)
optimizer = optim.Adam(regressor.parameters(), learning_rate)
criterion = torch.nn.MSELoss()

# Training and Testing data subdivision and control.
ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)
ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True)


### TRAINING PHASE ###

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
	# refer to the WandB data files for a better understanding
	#print(loss)

	# prints out the data for each iteration
	if wandB_control:
		wandb.log({'epoch': epoch, 'loss': loss})


### VALIDATION PHASE ###
# deactivates the net's ability to be trained: enter testing phase
def validation(X_test, y_test, regressor):
	regressor.eval()
	with torch.no_grad():

		# Same principles apply here
	    pred = regressor(X_test)
	    loss = criterion(pred, y_test)
	    print("========================================================")
	    print("testing loss:   ", loss)
	    
	    if wandB_control:
	    	wandb.log({'testing loss': loss})

### ACTIVATION FUNCTION FOR VALIDATION COMPARISON ###
if validTest_control:
	validation(X_test, y_test, regressor)


### ITERATIVE TOOL ###
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
	ci_interval = ci_upper - ci_lower

	return means, stds, ci_upper, ci_lower, ci_interval


### AGGREGATE PERFORMANCE REVIEW ###
# we seek to understand, on the whole system, how the estimated values differ
# from known values using standard deviation in the process
def master_testing(regressor, dataset, X, y, samples):

	# find the length of the dataset in totality, as well as refitting the 
	# original datapoint and target outputs in correct form
	DimSet = len(dataset)
	X, y = torch.tensor(X).float(), torch.tensor(y).float()

	# iterate through every entry of the datasheet
	for entry in range(DimSet):

		# create a Gaussian distribution through every data point
		prediction = [regressor(X[entry]) for gaussian_iteration in range(samples)]
		prediction = torch.stack(prediction)
		means = prediction.mean(axis=0)
		stds = prediction.std(axis=0)		

		# compare the estimated mean and the target value using standard
		# deviation. This will prove the effectiveness of the data
		dev_distance = (means - y[entry]) / stds

		# log data on to WandB
		if wandB_control:
			wandb.log({'deviation distance:': dev_distance})


### ACTIVATION FUNCTION FOR AGGREGATE PERFORMANCE REVIEW ###
# note: this should not be exercised frequently (time consuming)
if mast_Test_control:

	# activate the main function (this will take 10 minutes)
	master_testing(regressor, dataset, X, y, samples)


### INDIV. PERFORMANCE REVIEW ###
# in theory, an accurate model will yield an estimation that is same as the 
# actual target value in the dataset, thus we test a known point
def indiv_testing(regressor):

	# user input of data points
	cement          = input("cement                 (kg/m^3) [100-600]: 	")
	slag            = input("slag                   (kg/m^3) [000-400]: 	")
	ash             = input("ash                    (kg/m^3) [000-350]: 	")
	water           = input("water                  (kg/m^3) [100-300]: 	")
	sup_plasticizer = input("super-plasticizer      (kg/m^3) [000-050]: 	")
	c_Aggre         = input("coarse-aggregate       (kg/m^3) [500-1300]: 	")
	f_Aggre         = input("fine-aggregate         (kg/m^3) [500-1300]: 	")
	age             = input("age of sample          (day(s)) [001-365]: 	")

	X_response_test = [float(cement), float(slag), float(ash), float(water), 
					   float(sup_plasticizer), float(c_Aggre), float(f_Aggre), float(age)]
	X_response_test = torch.tensor(X_response_test).float()

	# tune parameter
	std_multiplier = 2

	# actually running the regression test
	means, stds, ci_upper, ci_lower, ci_interval = regression_test(regressor, 
										X_response_test, samples, std_multiplier)

	print("mean:            ", means)
	print("STD:             ", stds)
	print("upper CL:        ", ci_upper)
	print("lower CL:        ", ci_lower)
	print("confidence int:  ", ci_interval)


### ACTIVATION FUNCTION FOR INDIV. PERFORMANCE REVIEW ###
# if true, the system allows user inputs
# if false, the following lines are skipped entirely
while indivTest_control:

	indiv_testing(regressor)
	test_continuity = input("conduct another test? [Y]/[N]: ")
	print("========================================================")

	if test_continuity == "N":

		break

