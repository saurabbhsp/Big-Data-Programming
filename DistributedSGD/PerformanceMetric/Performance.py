import numpy as np
import math

from Logger.TimeIt import timeit

def SquaredError(yTrue, yPrediction):
	return np.sum((yTrue - yPrediction) ** 2)

def MSE(yTrue, yPrediction):
	return SquaredError(yTrue, yPrediction)/len(yTrue)

def RMSE(yTrue, yPrediction):
	return math.sqrt(MSE(yTrue, yPrediction))

@timeit 
def distributedMSE(root, communicator, yTrue, yPrediction):
	localSquaredError = SquaredError(yTrue, yPrediction)
	assimilatedSquaredError = communicator.gather([localSquaredError, len(yTrue)])
	if communicator.Get_rank() == root:
		globalSquaredSum = 0
		globalCount = 0
		for localSquaredError in assimilatedSquaredError:
			globalSquaredSum = globalSquaredSum + localSquaredError[0]
			globalCount = globalCount + localSquaredError[1]
		return (globalSquaredSum/globalCount)


@timeit
def distributedRMSE(root, communicator, yTrue, yPrediction):
	globalMSE = distributedMSE(root, communicator, yTrue, yPrediction)
	if communicator.Get_rank() == root:
		return math.sqrt(globalMSE)