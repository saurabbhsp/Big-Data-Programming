import numpy as np

from random import shuffle
from Logger.TimeIt import timeit

"""
This module includes all the methods
for data preprocessing
"""

"""Test set size should be percentage"""
@timeit
def splitDataSet(data, testSetSize):
	testSetSplit = round(len(data)*testSetSize)
	shuffle(data)
	return data[testSetSplit:], data[:testSetSplit]

@timeit
def distributedMaxScaler(data, communicator = None, root = None):
	maximum = data.max(axis = 0)

	if communicator != None:
		globalMaximum = communicator.gather(maximum, root)
		if communicator.Get_rank() == root:
			globalMaximum = np.vstack(globalMaximum)
			globalMaximum = globalMaximum.max(axis = 0).reshape(1, -1)
		maximum = communicator.bcast(globalMaximum, root)

	scaledData = (data - maximum)
	return scaledData


@timeit
def distributedMinMaxScaler(data, communicator = None, root = None):
	minimum = data.min(axis = 0)
	maximum = data.max(axis = 0)
	if communicator != None:
		globalMinimum = communicator.gather(minimum, root)
		globalMaximum = communicator.gather(maximum, root)
		if communicator.Get_rank() == root:
			globalMaximum = np.vstack(globalMaximum)
			globalMaximum = globalMaximum.max(axis = 0).reshape(1, -1)
			globalMinimum = np.vstack(globalMinimum)
			globalMinimum = globalMinimum.min(axis = 0).reshape(1, -1)
		maximum = communicator.bcast(globalMaximum, root)
		minimum = communicator.bcast(globalMinimum, root)

	scaledData = (data - minimum)/(maximum - minimum + 1)
	return scaledData
