import numpy as np
import math

from mpi4py import MPI
from random import shuffle

from Logger.TimeIt import timeit
from PerformanceMetric import Performance

"""
prediction = XB
"""
def prediction(X, B):
	return np.dot(X, B)

"""The RSS is defined as
RSS  = (y - yPrediction)^2

X = (n, m)
Y = (n, 1)
B = (m, 1)

where y is  a vector of dimension n X 1
B are the weights of dimension 1 X m
and X is of dimension n X m
where m is no of features and n is total no of samples

The gradient of the RSS is 
-X.T(y - yprediction)
"""
def getGradient(X, Y, B):
	residual = Y - prediction(X, B)
	return (-1) * np.dot(X.T, residual)


def L2Regularization(regularizationParameter, penalty):
	loss = 2 * regularizationParameter * penalty
	"Remove impact on bias"
	loss[0] = 0
	return loss

"""Calculate stochastic gradient descent
, if a communicator is provided the method will
calculate PSGD else will calculate SGD

Additionally for faster computation for reshuffiling of data
train set and test set is not split, instead 
their indices are split"""
@timeit
def calculatePSGD(featureSet, targetSet, trainingIndices, beta, learningRate, maxEpochs,
	regularization = None, penalty = None, communicator = None, tolerance = 1.0e-10):
	prevRMSE = 0
	for i in range(0, maxEpochs):
		print("Epoch-"+str(i+1))
		"""shuffle data"""
		shuffle(trainingIndices)
		"""Perform gradient descent for each sample"""
		for trainingIndex in trainingIndices:
			x = featureSet[trainingIndex, None]
			y = targetSet[trainingIndex, None]
			gradient = getGradient(x, y, beta)
			if regularization != None:
				gradient = gradient + regularization(beta, penalty)
			beta = beta - (learningRate * gradient)

		if communicator != None:
			"""Get new gradient as average of centroids"""
			communicator.barrier()
			beta = communicator.allreduce(beta, op = MPI.SUM)
			beta = beta/communicator.Get_size()

		predictions = prediction(featureSet[trainingIndices], beta)
		RMSE = Performance.RMSE(targetSet[trainingIndices], predictions)

		if i%20 == 0:
			if communicator != None:
				"""Test convergence among all workers"""
				localConvergence = False
				if tolerance > math.fabs(prevRMSE - RMSE):
					localConvergence = True
					print("Converged locally")
				globalConvergenceList = communicator.allgather(localConvergence)
				communicator.barrier()
				globalConvergence = True
				for localConvergence in globalConvergenceList:
					if localConvergence == False:
						"""Atleast one worker has not converged"""
						globalConvergence = False
						break
				if globalConvergence == True:
					print("Converged globally")
					return beta
			else:
				"""Test convergence locally"""
				if tolerance > math.fabs(prevRMSE - RMSE):
					print("Converged")
					return beta
		prevRMSE = RMSE
		print("RMSE "+str(RMSE))
	return beta
