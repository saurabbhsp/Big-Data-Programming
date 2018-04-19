from mpi4py import MPI
import sys
import numpy as np

"""Create global communicator"""
comm = MPI.COMM_WORLD

"""Following two procedures will simulate scatter and gather operation.
For this problem we will only use MPI point to point communication : Send and Receive"""


def scatter(data, root):

    """Only the root node will send the data to all the processors"""
    if comm.Get_rank() == root:

        totalProcessors = comm.Get_size()
        dataChunks = np.array_split(data, totalProcessors)
        rootChunk = None
        for processorRank in range(0, totalProcessors):

            if processorRank == root:
                rootChunk = dataChunks[processorRank]
            else:
                comm.send(dataChunks[processorRank], dest=processorRank)
                # comm.Isend(dataChunks[processorRank], dest=processorRank)
        return rootChunk

    """All other nodes will get the data"""
    # request = comm.Irecv(np.zeros(1), source=root)
    data = comm.recv(source = root)
    # request.Wait()
    return data

def gather(data, root):

    """All except root will send the data"""

    if comm.Get_rank() != root:
        comm.send(data, dest = root)
        return None

    
    """The data joining will occur at root only"""
    if comm.Get_rank() == root:

        totalProcessors = comm.Get_size()
        outputData = np.array([])

        for processorRank in range(0, totalProcessors):

            if processorRank == root:
                outputData = np.append(outputData, data)
            else:
                dataRecv = comm.recv(source = processorRank)
                outputData = np.append(outputData, dataRecv)
        return outputData


"""Store the total processes and current process rank. 
The process 0 will be considered as master process and will store the final output"""

size = comm.Get_size()
currentRank = comm.Get_rank()

"""Given a vector A, we have to calculate the average of it"""
vectorA = None
"""We will consider the root as 0. The random vector will only be generated on the root vector.
Alternatively we can also read the vector for file system on independent processors"""

root = 0

globalStartTime = MPI.Wtime()

"""Generate random vector on root"""
if currentRank == root:
	dataGenerationStarted = MPI.Wtime()
	vectorA = np.random.randint(100, size=int(eval(sys.argv[1])))
	print("Vector initialized")
	print(vectorA)
	"""COMMENT FOLLOWING LINE TO PREVENT STORAGE OF VECTOR TO DISK"""
	#np.savetxt("VectorA", vectorA)
	dataGenerationEnded = MPI.Wtime()
	print("-->Time taken to generate data = "+ str(dataGenerationEnded - dataGenerationStarted))


"""Following code will run on all the processors in parallel
 Big vector will be scattered accross all the processors
 Then each processor will add the small vector and send it back to the root"""

vecSum = np.sum(scatter(vectorA, root))
assimilatedVector = gather(vecSum, root)

if currentRank == root:
	bigSum = np.sum(assimilatedVector)
	average = (bigSum*1.0)/int(eval(sys.argv[1]))
	print("The average of the vector is "+str(average))

endTime = MPI.Wtime()

if currentRank != root:
    print("-->Time taken on processor "+ str(currentRank) + " - "+ str(endTime - globalStartTime))

else:
    print("-->Time taken on processor "+ str(currentRank) + " - "+ str((endTime - globalStartTime) - (dataGenerationEnded - dataGenerationStarted)))
    print("-->Total time taken including data generation "+ str(currentRank) + " - "+ str(endTime - globalStartTime))
