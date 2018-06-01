import sys
import numpy as np
import os
import pandas as pd

from mpi4py import MPI
from Utils import Preprocessing
from PerformanceMetric import Performance
from Regression import CD

"""Command line arguments"""
inputPath = sys.argv[1]
outputPath = sys.argv[2]
maxEpochs = int(sys.argv[3])

"""MPI variables"""
comm = MPI.COMM_WORLD
processorCount = comm.Get_size()
currentRank = comm.Get_rank()
root = 0


"""For storing the assigned task to the processor"""
processorTasks = []
xTrain = None
yTrain = None
xTest = None
yTest = None


if currentRank == root:
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	trainData = pd.read_csv(os.path.join(inputPath, "Train", "cup98LRN.txt"))
	"""Selected Features"""
	featureList = ['STATE', 'PVASTATE', 'MDMAUD', 'CLUSTER', 'GENDER', 'HIT',
	'DATASRCE', 'MALEMILI', 'MALEMILI', 'VIETVETS', 'WWIIVETS', 'LOCALGOV', 'STATEGOV', 
	'FEDGOV','CARDPROM', 'NUMPROM', 'RAMNTALL', 'NGIFTALL', 'CARDGIFT', 'AVGGIFT', 'TARGET_D']
	trainData = trainData[featureList]
	trainData = pd.get_dummies(trainData)
	processorTasks = np.array_split(trainData, processorCount)

trainData = comm.scatter(processorTasks, root)

"""Feature cleanup"""
features = np.array(trainData.loc[:, trainData.columns != 'TARGET_D'])
target = np.array(trainData['TARGET_D']).reshape(-1, 1)


"""Split in train and test set"""
totalData = list(range(0, len(trainData)))
trainSetIndices, testSetIndices = Preprocessing.splitDataSet(totalData, 0.3)


"""Min max scaler"""
features = Preprocessing.distributedMinMaxScaler(features, comm, root)

"""Add bias""" 
features = np.insert(features, -1, 1, axis = 1)


modelParams = None
if currentRank == root:
	modelParams = np.random.uniform(1, 5, (features.shape[1], 1))


modelParams = comm.bcast(modelParams, root)


startTime = MPI.Wtime()
"""Perform coordinate descent"""
CD.distributedCoordinateDescent(features[trainSetIndices], target[trainSetIndices], 0.9, modelParams, maxEpochs, comm)


endTime = MPI.Wtime()
print(endTime - startTime)