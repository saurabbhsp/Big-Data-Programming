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

"""Vectors for storing the values for addition.
C = A + B 
Currently vectors A and B will be initialized on the master, however it is possible to read these on master and 
scatter on all processes, on the other hand the same can be read from a file on each host systems
"""
vectorA = None
vectorB = None
vectorC = None

root = 0

globalStartTime = MPI.Wtime()
"""Only master will print"""

if currentRank == root:
    dataGenerationStarted = MPI.Wtime()
    print("Creating the vectors of size " + str(sys.argv[1]) + " and total processes spawned "+str(size))

    """Initialize the random vectors"""
    vectorA = np.random.randint(100, size=int(eval(sys.argv[1])))
    vectorB = np.random.randint(100, size=int(eval(sys.argv[1])))

    print("Vector initialized are")
    print(vectorA)
    print(vectorB)

    """Save vectors"""
    """COMMENT FOLLOWING LINE TO PREVENT STORAGE OF VECTOR TO DISK"""
    #np.savetxt("VectorA", vectorA)
    #np.savetxt("VectorB", vectorB)

    dataGenerationEnded = MPI.Wtime()

    print("-->Time taken to generate data = "+ str(dataGenerationEnded - dataGenerationStarted))
"""Following code will run in parallel
1) Split the vectors on all the processors
2) Add the vectors. Addition will run in parallel
3) Gather the data back on the root"""

vecA = scatter(vectorA, root)
vecB = scatter(vectorB, root)
vecC = vecA + vecB

vectorC = gather(vecC, root)

if currentRank == root:
    print("Output of addition of two vectors")
    print(vectorC)

    """Save vector output"""
    #np.savetxt("VectorC", vectorC)

endTime = MPI.Wtime()

if currentRank != root:
    print("-->Time taken on processor "+ str(currentRank) + " - "+ str(endTime - globalStartTime))

else:
    print("-->Time taken on processor "+ str(currentRank) + " - "+ str((endTime - globalStartTime) - (dataGenerationEnded - dataGenerationStarted)))
    print("-->Total time taken including data generation "+ str(currentRank) + " - "+ str(endTime - globalStartTime))

