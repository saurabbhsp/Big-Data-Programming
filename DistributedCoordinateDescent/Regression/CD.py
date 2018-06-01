import numpy as np
import math

from Logger.TimeIt import timeit
from mpi4py import MPI
from PerformanceMetric import Performance

"""
prediction = XB
"""
def prediction(X, B):
	return np.dot(X, B)


"""
Data dimensions

X = (n, m)
Y = (n, 1)
B = (m, 1)
"""
@timeit
def distributedCoordinateDescent(trainSetX, trainSetY, alpha, beta, maxEpochs, communicator = None, tolerance = 1.0e-10):
	cache = {}
	yPrediction = prediction(trainSetX, beta)
	prevRMSE = 0

	for i in range(0, maxEpochs):
		for coordinate in range(beta.shape[0]):

			denominator = 0
			featureVector = trainSetX[:, coordinate].reshape(-1, 1)
			if coordinate not in cache:
				denominator =  np.dot(featureVector.T, featureVector)
				cache[coordinate] = denominator
			else:
				denominator = cache[coordinate]


			numerator = np.dot(featureVector.T, (trainSetY - yPrediction + featureVector * beta[coordinate]))

			if communicator != None:
				communicator.barrier()
				aggregate = communicator.allreduce([numerator, denominator], op = MPI.SUM)
				numerator = aggregate[0]/communicator.Get_size()
				denominator = aggregate[1]/communicator.Get_size()

			betaOld = beta[coordinate][0]

			beta[coordinate][0] = (1 - alpha) * beta[coordinate] +  alpha * (numerator/(denominator + 0.001))

			yPrediction = yPrediction + featureVector * (beta[coordinate][0] - betaOld)


		RMSE = Performance.RMSE(trainSetY, yPrediction)
		if i%20 == 0 and communicator != None:
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
		elif i%20==0:
			"""Test convergence locally"""
			if tolerance > math.fabs(prevRMSE - RMSE):
				print("Converged")
				return beta

		print(str(i)+ " "+ str(RMSE))
		prevRMSE = RMSE
		print("Epoch completed")
	return beta