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

# wandb sweep Bayesian_Optimization.yaml

##################################################################################
### CONTROL PANEL ###
# procedure and module control
wandB_control =         True
validTest_control =     True
# no serious use without WandB being True
melb_Test_control =     True
lond_Test_control = 	True
indivTest_control =     False 

##################################################################################
### HYPERPARAMETER LISTING ###
hyperparameter_defaults = dict(
	                           # untuned  tuned
	blayer1 = 128,             # 128      292
	blayer2 = 32,              # 32       232
	batch_size = 1,            # 1        1
	learning_rate = 0.001,     # 0.001    0.0155
	EPOCHS = 31,               # 31       31
	samples = 200              # 200      200
	)
# WandB documentation
if wandB_control:
	
	wandb.init(config = hyperparameter_defaults, project="Bayesian_Engine.py")
	config = wandb.config

	# the hyperparameters will now be configurable be the yaml file
	blayer1 = config.blayer1
	blayer2 = config.blayer2
	batch_size = config.batch_size
	learning_rate = config.learning_rate
	EPOCHS = config.EPOCHS
	samples = config.samples

if wandB_control != True:

	# the hyperparameters will not be changed by the yaml file
	blayer1 = hyperparameter_defaults["blayer1"]
	blayer2 = hyperparameter_defaults["blayer2"]
	batch_size = hyperparameter_defaults["batch_size"]
	learning_rate = hyperparameter_defaults["learning_rate"]
	EPOCHS = hyperparameter_defaults["EPOCHS"]
	samples = hyperparameter_defaults["samples"]

##################################################################################
### DATA MANIPULATION ###
# access the base dataset
datasetMel = pd.read_csv('./Mel.csv')

# some data manipulation:
# extract the data out of the CSV file
mel_X = datasetMel.iloc[:, 0:8].values
mel_y = datasetMel.iloc[:, 8:9].values
# normalizing the data (currently not in use)
#X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y)

# divide the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mel_X, mel_y, test_size=.25, random_state=42)
X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

datasetLon = pd.read_csv('./UCL.csv')
lon_W = datasetLon.iloc[:, 0:8].values
lon_z = datasetLon.iloc[:, 8:9].values

##################################################################################
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

##################################################################################
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

##################################################################################
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

##################################################################################
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

##################################################################################
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

##################################################################################
### MELBOURNE AGGREGATE PERFORMANCE REVIEW ###
# we seek to understand, on the whole system, how the estimated values differ
# from known values using standard deviation in the process
def melb_master_testing(regressor, datasetMel, mel_X, mel_y, samples):

	# find the length of the dataset in totality, as well as refitting the 
	# original datapoint and target outputs in correct form
	mel_DimSet = len(datasetMel)
	mel_X, mel_y = torch.tensor(mel_X).float(), torch.tensor(mel_y).float()

	# iterate through every entry of the datasheet
	for entry in range(mel_DimSet):

		# create a Gaussian distribution through every data point
		prediction = [regressor(mel_X[entry]) for gaussian_iteration in range(samples)]
		prediction = torch.stack(prediction)
		means = prediction.mean(axis=0)
		stds = prediction.std(axis=0)		

		# compare the estimated mean and the target value using standard
		# deviation. This will prove the effectiveness of the data
		dev_distance_Mel = (means - mel_y[entry]) / stds
		
		# NOTE: entry 1 [336, 0, 0, 182, 3, 986, 817, 28] is entry two 
		#		of the .CSV file. Always remember to plus 1 on the translation

		# log data on to WandB
		if wandB_control:
			wandb.log({'entry stamp': entry, 'Deviation Distance Melbourne:': dev_distance_Mel})

##################################################################################
### ACTIVATION FUNCTION FOR AGGREGATE PERFORMANCE REVIEW ###
# note: this should not be exercised frequently (time consuming)
if melb_Test_control:

	# activate the main function (this will take 10 minutes)
	melb_master_testing(regressor, datasetMel, mel_X, mel_y, samples)

##################################################################################
### UCL AGGREGATE PERFORMANCE TEST ###
# Now that it seems that the machine has completed its training on MEL.csv
# can this regressor be used on UCL.csv, would the regressor operate with
# the same loss capacity?
def lond_master_testing(regressor, datasetLon, lon_W, lon_z, samples):

	# find the length of the dataset in totality, as well as refitting the 
	# original datapoint and target outputs in correct form
	lon_DimSet = len(datasetLon)
	lon_W, lon_z = torch.tensor(lon_W).float(), torch.tensor(lon_z).float()

	# iterate through every entry of the datasheet
	for entry in range(lon_DimSet):

		# create a Gaussian distribution through every data point
		prediction = [regressor(lon_W[entry]) for gaussian_iteration in range(samples)]
		prediction = torch.stack(prediction)
		means = prediction.mean(axis=0)
		stds = prediction.std(axis=0)

		# compare the estimated mean and the target value using standard
		# deviation. This will prove the effectiveness of the data
		dev_distance_Lon = (means - lon_z[entry]) / stds
		
		# NOTE: entry 1 [336, 0, 0, 182, 3, 986, 817, 28] is entry two 
		#		of the .CSV file. Always remember to plus 1 on the translation

		# log data on to WandB
		if wandB_control:
			wandb.log({'entry stamp': entry, 'Deviation Distance London:': dev_distance_Lon})

##################################################################################
### ACTIVATION FUNCTION FOR AGGREGATE PERFORMANCE REVIEW ###
# note: this should not be exercised frequently (time consuming)
if lond_Test_control:

	# activate the main function (this will take 10 minutes)
	lond_master_testing(regressor, datasetLon, lon_W, lon_z, samples)	

##################################################################################
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

	# actually running the regression test
	means, stds, ci_upper, ci_lower, ci_interval = regression_test(regressor, 
										X_response_test, samples, std_multiplier = 2)

	print("mean:            ", means)
	print("STD:             ", stds)
	print("upper CL:        ", ci_upper)
	print("lower CL:        ", ci_lower)
	print("confidence int:  ", ci_interval)

##################################################################################
### ACTIVATION FUNCTION FOR INDIV. PERFORMANCE REVIEW ###
# if true, the system allows user inputs
# if false, the following lines are skipped entirely
while indivTest_control:

	indiv_testing(regressor)
	test_continuity = input("conduct another test? [Y]/[N]: ")
	print("========================================================")

	if test_continuity == "N":

		break

