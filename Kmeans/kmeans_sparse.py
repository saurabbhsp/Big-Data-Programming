import os
import sys
import numpy as np
import random
import scipy
import json
import matplotlib.pyplot as plt
import _pickle as pickle
import cProfile

from mpi4py import MPI
from os.path import basename, join
from random import shuffle
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix

def buildInput(basePath):
	pathList = []
	for file in os.listdir(basePath):
		pathList.append(join(basePath, file))
	return pathList

def writeToFile(path, text):
	with open(path, 'w') as outputFile:
		outputFile.write(text)

def readJSONFile(path):
	outputJSON = None
	with open(path, 'r') as inputFile:
		outputJSON = json.load(inputFile)
	return outputJSON

"""Will return distance from all clusters"""
def getDistanceFromCluster(clusterCenters, dataMatrix):
	distanceList = []

	"""Calculate each cluster centroid
	distance from all data points in data matrix"""
	for clusterCenter in clusterCenters:
		distanceList.append(euclidean_distances(clusterCenter, dataMatrix).flatten())
	return np.array(distanceList)

def getFarthestPoint(clusterCenters, dataMatrix):
	clusterDistance = getDistanceFromCluster(clusterCenters, dataMatrix)
	clusterDistance = clusterDistance.sum(axis = 0)
	"""Find the index with max distance"""
	index = np.where(clusterDistance == max(clusterDistance))[0][0]
	return dataMatrix.getrow(index), max(clusterDistance)

def jsonToCSRMatrix(pathList, dimensions):
	data = []
	rows = []
	columns = []
	for i in range(0, dimensions[0]):
		jsonDoc = readJSONFile(pathList[i])
		for key, value in jsonDoc.items():
			"""Convert the data to float16 to reduce memory footprint"""
			data.append(np.float16(value))
			columns.append(int(key))
			rows.append(i)
	matrix = csr_matrix((data, (rows, columns)), shape=dimensions)
	return matrix

def assignClusters(clusterCenters, dataMatrix):
	distortion = 0
	clusterMembership = {}
	"""Get distance of all points from all clusters."""
	clusterDistance = getDistanceFromCluster(clusterCenters, dataMatrix)
	"""Now each row is array having list of distances from cluster centers"""
	clusterDistance = clusterDistance.T

	for dataPointIndex, dataPointDistance in enumerate(clusterDistance):
		minDistance = min(dataPointDistance)
		clusterIndex = np.where(dataPointDistance == minDistance)[0][0]
		distortion = distortion + minDistance
		if clusterIndex not in clusterMembership:
			clusterMembership[clusterIndex] = []
		"""The datapoint index is local to the processor"""
		clusterMembership[clusterIndex].append(dataPointIndex)
	return clusterMembership, distortion

def recenter(clusterMembership, dataMatrix):
	newCenters = {}
	for clusterIndex in clusterMembership:
		subMatrix = dataMatrix[clusterMembership[clusterIndex]]
		"""Tuple of cumulative sum of 
		vectors of cluster members and total count of cluster members"""
		newCenters[clusterIndex] = [csr_matrix(csr_matrix.sum(subMatrix, axis = 0)), len(clusterMembership[clusterIndex])]
	return newCenters

def recenterGlobalClusterCenters(globalClusterMembership):
	"""This method will calculate global cluster centroids
	using previously calculated local cluster centroid"""
	cummulativeCenters = {}
	cummulativeCount = {}
	clusterCenters = []

	for clusterMembership in globalClusterMembership:
		for clusterIndex in clusterMembership:
			if clusterIndex in newCenters:
				cummulativeCenters[clusterIndex] = cummulativeCenters[clusterIndex] + clusterMembership[clusterIndex][0]
				cummulativeCount[clusterIndex] = cummulativeCount[clusterIndex] + clusterMembership[clusterIndex][1]
			else:
				cummulativeCount[clusterIndex] = clusterMembership[clusterIndex][1]
				cummulativeCenters[clusterIndex] = clusterMembership[clusterIndex][0]
	"""Calculate global cluster mean and 
	store it in new list"""
	for i in range(0, len(cummulativeCenters)):
		clusterCenters.append(cummulativeCenters[i]/cummulativeCount[i])
	return clusterCenters

def localClusterLookup(clusterMembership, dataPointLookup):
	"""For Index to data point mapping"""
	clusterMembers = {}
	for clusterIndex in clusterMembership:
		clusterMembers[clusterIndex] = [dataPointLookup[dataPointIndex] for dataPointIndex in clusterMembership[clusterIndex]]
	return clusterMembers

def globalClusterLookup(localClusterMembers, clusterCenters):
	globalClusterMembers = {}
	for localClusterMember in localClusterMembers:
		for clusterIndex in localClusterMember:
			if str(clusterIndex) not in globalClusterMembers:
				globalClusterMembers[str(clusterIndex)] = []
			globalClusterMembers[str(clusterIndex)].extend(localClusterMember[clusterIndex])
	return globalClusterMembers

"""Command line arguments"""
inputPath = sys.argv[1]
outputPath = sys.argv[2]
k = int(sys.argv[3])
maxEpochs = int(sys.argv[4])

"""MPI variables"""
comm = MPI.COMM_WORLD
processorCount = comm.Get_size()
currentRank = comm.Get_rank()
root = 0


processorTasks = None
dataPoints = {}
clusterCenters = []

vocabSize = None

"""Build the list of inputs"""
if currentRank == root:
	startTime = MPI.Wtime()
	pathList = buildInput(join(inputPath, "TFIDF"))
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	shuffle(pathList)
	metaData = readJSONFile(join(inputPath, "metadata.json"))
	vocabSize = metaData["vocabLength"]
	vocabSize = int(vocabSize)
	processorTasks = np.array_split(pathList, processorCount)

vocabSize = comm.bcast(vocabSize, root)
processorTasks = comm.scatter(processorTasks, root)


"""For faster computation all processors will build a CSR matrix"""
dataMatrix = jsonToCSRMatrix(processorTasks, (len(processorTasks), vocabSize))

"""Used for storing the name of file. Useful for identifying which item belongs to which cluster"""
dataPointLookup = []
for processorTask in processorTasks:
	dataPointLookup.append(basename(processorTask))


"""First center will be selected from random from data available with root"""
if currentRank == root:
	randomIndex = random.randint(0, dataMatrix.get_shape()[0])
	clusterCenters.append(dataMatrix.getrow(randomIndex))
	print("Cluster center-1 calculated")

clusterCenters = comm.bcast(clusterCenters, root)

"""Next clusters will be selected as one having farthest distance"""
for i in range(1, k):
	farthestPoint, farthestDistance = getFarthestPoint(clusterCenters, dataMatrix)

	clusterCenterCandidates = comm.gather([farthestDistance, farthestPoint], root)

	if currentRank == root:
		maxDistance = 0
		maxDistanceDataPoint = None

		for clusterCenterCandidate in clusterCenterCandidates:
			if maxDistance < clusterCenterCandidate[0]:
				maxDistance = clusterCenterCandidate[0]
				maxDistanceDataPoint = clusterCenterCandidate[1]
		clusterCenters.append(maxDistanceDataPoint)
		print("Cluster center-"+str(i+1)+" calculated")

	"""BroadCast the updated cluster centers to all"""
	clusterCenters = comm.bcast(clusterCenters, root)

"""New cluster centers are now available at each node"""


"""Start clustering process"""
currentEpoch = 0
prevClusterMembership = {}
distortions = []
while(currentEpoch < maxEpochs):

	if currentRank == root:
		print("Epoch-"+str(currentEpoch+1))

	"""Step-1 Assign data point to cluster centers"""
	clusterMembership, distortion = assignClusters(clusterCenters, dataMatrix)
	totalDistortion = comm.gather(distortion, root)
	if currentRank == root:
		totalDistortion = np.sum(totalDistortion)
		print("Distortion is "+str(totalDistortion))
		distortions.append(totalDistortion)

	"""Check if clusters changed locally"""
	if prevClusterMembership == clusterMembership:
		clusterMembershipChanged = False
	else:
		clusterMembershipChanged = True

	prevClusterMembership = clusterMembership

	"""Check if clusters are changed globally"""
	comm.barrier()
	globalClusterMembershipChanged = comm.allgather(clusterMembershipChanged)
	convergenceStatus = True
	for globalUpdateStatus in globalClusterMembershipChanged:
		if globalUpdateStatus == True:
			"""Atleast one processor cluster members changed"""
			convergenceStatus = False


	if convergenceStatus == True:
		"""No change in cluster membership on any worker.
		Kmeans  has converged"""
		if currentRank == root:
			print("Kmeans converged")
		break

	"""No convergence calculate new cluster center"""
	"""Step-2 Calculate new cluster centroids"""
	newCenters = recenter(clusterMembership, dataMatrix)
	newCenters = comm.gather(newCenters, root)
	if currentRank == root:
		clusterCenters = recenterGlobalClusterCenters(newCenters)

	comm.barrier()
	clusterCenters = comm.bcast(clusterCenters, root)
	currentEpoch+=1

"""End of while loop"""

"""Calculate and store cluster membership
All cluster memberships are indices to dataMatrix. 
Cluster membership will be calculated using lookup"""

localClusterMembers = localClusterLookup(clusterMembership, dataPointLookup)
globalClusterMembers = comm.gather(localClusterMembers, root)

if currentRank == root:
	"""Merge local processors lookup """
	globalClusterMembers = globalClusterLookup(globalClusterMembers, clusterCenters)
	writeToFile(join(outputPath, "clusterMembers.json"), json.dumps(globalClusterMembers, indent=2))

	for index, clusterCenter in enumerate(clusterCenters):
		pickle.dump(clusterCenter, open(join(outputPath, str(index)+".pickle"), 'wb'))

	"""Save plots"""
	plt.plot(range(0, len(distortions)), distortions)
	plt.plot(range(0, len(distortions)), distortions, "rx")
	plt.xticks(range(0, len(distortions)))
	plt.title("Iteration vs distortion")
	plt.xlabel("Iteration")
	plt.ylabel("Distortion")
	plt.savefig(join(outputPath, "iterationsVSdistortion.png"))
	plt.clf()
	distortions = distortions[1:]
	plt.plot(range(0, len(distortions)), distortions)
	plt.plot(range(0, len(distortions)), distortions, "rx")
	plt.xticks(range(0, len(distortions)))
	plt.title("Iteration vs distortion")
	plt.xlabel("Iteration")
	plt.ylabel("Distortion")
	plt.savefig(join(outputPath, "iterationsVSdistortion2.png"))

	endTime = MPI.Wtime()
	print("Ran for "+str(currentEpoch)+" iterations")
	print("Total time taken "+str(endTime - startTime))
