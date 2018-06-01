import sys
import numpy as np
import os

from mpi4py import MPI
from Utils import DataReader
from Utils import Preprocessing
from Regression import SGD
from PerformanceMetric import Performance

"""Command line arguments"""
inputPath = sys.argv[1]
outputPath = sys.argv[2]
maxEpochs = int(sys.argv[3])

featureCount = 482

"""MPI variables"""
comm = MPI.COMM_WORLD
processorCount = comm.Get_size()
currentRank = comm.Get_rank()
root = 0


"""For storing the assigned task to the processor"""
processorTasks = []

if currentRank == root:
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	processorTasks = np.array_split(DataReader.listAllFiles(inputPath), processorCount)

processorTasks = comm.scatter(processorTasks, root)
features, target = DataReader.svmLightToNPVectors(processorTasks, featureCount ,np.float32, True)

features = features.toarray().reshape(-1, featureCount + 1)
"""Split data set into train and test set locally.
Each processor has its own copy of train set and
test set"""
totalData = list(range(0, len(target)))
trainSetIndices, testSetIndices = Preprocessing.splitDataSet(totalData, 0.3)


modelParams = None
if currentRank == root:
	#modelParams = np.random.uniform(-1e-3, 1e-3, (featureCount + 1, 1))
	modelParams = np.random.uniform(-1e-4, 1e-4, (featureCount + 1, 1))


modelParams = comm.bcast(modelParams, root)



"""Min max scaler"""
features = Preprocessing.distributedMaxScaler(features, comm, root)

#modelParams = SGD.calculatePSGD(features, target, trainSetIndices, modelParams, 
#	1e-10, maxEpochs, communicator = comm)
startTime = MPI.Wtime()
modelParams = SGD.calculatePSGD(features, target, trainSetIndices, modelParams, 
	1e-12, maxEpochs, communicator = comm)
endTime = MPI.Wtime()
print(endTime - startTime)

predictions = SGD.prediction(features[testSetIndices], modelParams)

globalTestMSE = Performance.distributedMSE(root, comm, target[testSetIndices], predictions)
globalTestRMSE = Performance.distributedRMSE(root, comm, target[testSetIndices], predictions)

predictions = SGD.prediction(features[trainSetIndices], modelParams)
globalTrainMSE = Performance.distributedMSE(root, comm, target[trainSetIndices], predictions)
globalTrainRMSE = Performance.distributedRMSE(root, comm, target[trainSetIndices], predictions)

if currentRank == root:
	print("Total Train set RMSE is "+str(globalTrainRMSE))
	print("Total Test set RMSE is "+str(globalTestRMSE))
