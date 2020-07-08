import numpy as np 
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

d = 1

# git add Gaussian_Process.py
# git commit -m ""
# git remote add origin https://github.com/SaintJohn-Royce/2020_CSProject_ML.git
# git push -u origin master

### DATA MANIPULATION ###
# access the base dataset
dataset = pd.read_csv('./Mel.csv')

X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values

# 849 train points, 284 test points
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0010, random_state=42)

### CONVENTIONAL KERNEL FUNCTION ###
def RBF_Kernel(X1, X2, sigma_f, lengthScale):
	X1 = np.array(X1)
	X2 = np.array(X2)
	euclideanDistance = np.linalg.norm(X1-X2)
	kernel = (sigma_f**2) * np.exp(-1 * (euclideanDistance**2)/(2 * lengthScale**2))
	return kernel

sigma_f = 10
lengthScale = 200

### COVARIANCE MATRICES ###
def covariance_Matrices_Calculator(X_train, X_test, sigma_f, lengthScale):
	
	# Compute the components of the covariance matrix of the joint distribution
	# NOTE: "_test" = "star"
	#	K = K(X_train, X_train)				the "training" matrix
	# 	K_star = K(X_test, X_train)			the "train-test" matrix
	#	K_star2 = K(X_test, X_test)			the "testing" matrix

	# tune the datasets first
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	# find the dimensions for the training^2 matrix
	# basically the depth of X_train; the amount of datapointsin X_train
	trainingDim = X_train.shape[0]
	# same principles apply here in the case of testing data
	testingDim = X_test.shape[0]
	#print(trainingDim, testingDim)

	# build the training matrix
	K = [RBF_Kernel(i, j, sigma_f, lengthScale) for (i, j) 
		 in itertools.product(X_train, X_train)]
	K = np.array(K).reshape(trainingDim, trainingDim)

	# build the train-test matrix
	K_star = [RBF_Kernel(i, j, sigma_f, lengthScale) for (i, j)
			  in itertools.product(X_train, X_test)]
	K_star = np.array(K_star).reshape(trainingDim, testingDim)

	# build the testing matrix
	K_star2 = [RBF_Kernel(i, j, sigma_f, lengthScale) for (i, j)
			   in itertools.product(X_test, X_test)]
	K_star2 = np.array(K_star2).reshape(testingDim, testingDim)

	return (K, K_star, K_star2)

K, K_star, K_star2 = covariance_Matrices_Calculator(X_train, 
					 X_test, sigma_f, lengthScale)
combined_matrix_A = np.concatenate((K, K_star.T), axis=0)
combined_matrix_B = np.concatenate((K_star, K_star2), axis=0)
combined_matrix = np.concatenate((combined_matrix_A, combined_matrix_B), axis=1)

plt.imshow(combined_matrix)
plt.show()


def compute_gpr_parameters(K, K_star, K_star2, sigma_f, d):

	n = K.shape[0]

	# mean calculation
	K_noise = np.linalg.inv(K + sigma_f**2 * np.eye(n))
	f_bar = np.dot(K_noise, K_star)
	f_bar_star = np.dot(y_train.reshape([d, n]), f_bar)

	# covariance calculation
	cov_p1 = np.dot(K_star.T, K_noise)
	cov_f_star = K_star2 - np.dot(cov_p1, K_star)

	return f_bar_star, cov_f_star

f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star, K_star2, sigma_f, d)

plt.imshow(cov_f_star)
plt.show()

for i in range(0, 100):

	z_star = np.random.multivariate_normal(mean=f_bar_star.squeeze(), cov=cov_f_star)

	plt.plot(z_star, 'b')

plt.plot(y_test, 'r')

plt.show()



