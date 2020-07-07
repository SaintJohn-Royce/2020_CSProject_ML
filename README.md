# 2020_CSProject_ML
# Probabilistic Estimation of Material Strength using Machine Learning Methods

Concrete's material characteristics depends largely on the anisotropy of its solid components. Without anisotropy, concrete's material behaviour, namely concrete strength, will become unpredictable. As THE foundational material of the modern world, it is essential that concrete strength is controllable in order to put out a stable product. However, anisotropy is a variable that cannot be expressed as a unique class or value; this raises the difficulty of behaviour estimation as key constituents involved in concrete production may need to be ignored.

To combat this, Yeh (1998) utilized a Machine Learning (ML) method to "black box" this process; instead of proposing a model, he would have a machine to directly estimate an outcome from the data, as opposed to utilizing the Frequentist Approach. This was hugely promising as the accuracy he calculated was quite high, roughly standing at 85 to 95%. 

However, DeRousseau (2019), while supporting Yeh's formulation that ML should be the primary method of concrete material behaviour estimation, questioned the utility of Yeh's experiment as DeRousseau noted that the concrete samples in Yeh's model are often produced under closed environment, as opposed to being cast in-situ. DeRousseau backed up that claim by conducting said experiment on a batch of in-situ cast concrete, yielding an accuracy 30% lower than that of Yeh's.

Given the requirement of anisotry, and the lack of ability to measure said anisotropy, perhaps it would be better if a probabilstic method can be utilized. As opposed to a discrete estimation, probabilistic methods can yield a mean, standard deviation, and a confidence interval for each data point. This has potential as it distances itself from focusing on absolute confidence in a discrete point and instead opts to describe to the user the range of possible values that the estimation may be. From a jobsite perspective, a contractor will often be more comfortable with knowing a range of possible values for a concrete's strength, as opposed to blindly trusting, with full confidence, the strength the concrete is supposed to have. 

As of yet, this repository has several files:
  1. Mel.csv: the main dataset that is trained and tested on
  2. Discrete_Engine.py: the discrete ML method, akin to Yeh's method
  3. Bayesian_Engine.py: the probabilistic ML method, where the output is a Gaussian distribution

As of yet, the Discrete_Engine.py and the Bayesian_Engine.py are working smoothly (for now). As an academic interest, I would also propose the use of Gaussian Process Regression as another way of conducting probabilistic modelling, but that is still in the works.

The two titans of ML in the field of concrete behaviour, amongst many, are Yeh and DeRousseau. Below are links to their literature, respectively:

https://www.sciencedirect.com/science/article/abs/pii/S0008884698001653?via%3Dihub
https://www.sciencedirect.com/science/article/abs/pii/S0950061819320719

Lastly, I must thank Piero Esposito for building the Bayesian Layer (and offering it as a Github Repo!). Below is the main page I took inspiration from:

https://towardsdatascience.com/blitz-a-bayesian-neural-network-library-for-pytorch-82f9998916c7

I have also utilized WandB in hyperparameter tuning. A report on BNN tuning can be seen below:

https://app.wandb.ai/saintjohn-royce/A7/reports/Optimization-of-Project-2020_CSProject_ML%3A-Bayesian_Engine--VmlldzoxNjE5Mjc

July 1st, 2020
