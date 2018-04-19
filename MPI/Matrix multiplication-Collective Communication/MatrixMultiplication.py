from mpi4py import MPI
import sys
import numpy as np

"""Create global communicator"""
comm = MPI.COMM_WORLD

globalStartTime = MPI.Wtime()


"""Store the total processes and current process rank. 
The process 0 will be considered as master process and will store the final output"""

size = comm.Get_size()
currentRank = comm.Get_rank()
root = 0


"""A, B ,C are the three matrices. 
Vector dot product is a special case of matrix multiplication
C = A*B (Matrix dot product)
"""


matrixA = None
matrixB = None
matrixC = None


ARows = eval(sys.argv[1])
AColumns = eval(sys.argv[2])
BRows = eval(sys.argv[3])
BColumns = eval(sys.argv[4])

"""Matrix initialization will only happen on the root node. Then the data will be shared with other processors.
Alternatively it can also be read from file system by dividing blocks
(See Parallel matrix multiplication Tile method)"""


"""Following code will be executed only on root"""
if currentRank == root:
    dataGenerationStarted = MPI.Wtime()
    if AColumns != BRows:
        print("Matrix multiplication not possible with dimensons A["+str(ARows)+"]["+str(AColumns)+"] X B ["+str(BRows)+"]["+str(BColumns)+"]")

        """Will terminate all the processors"""
        comm.Abort()

    """Initialize the matrix randomly"""
    matrixA = np.random.randint(100, size=(ARows, AColumns))
    matrixB = np.random.randint(100, size=(BRows, BColumns))
    matrixC = np.zeros(shape=(ARows, BColumns))

    """COMMENT FOLLOWING LINE TO PREVENT STORAGE OF MATRIX TO DISK"""
    #np.savetxt("MatrixA", matrixA)
    #np.savetxt("MatrixB", matrixB)

    dataGenerationEnded = MPI.Wtime()
    print("-->Time taken to generate data = "+ str(dataGenerationEnded - dataGenerationStarted))


"""For special case where the input is vector. No need to resend the data again"""
scatterMatrixA = True
scatterMatrixB = True


"""Matrix dimesions are needed on all systems. Not actual data but just dimensions
Alternatively a message queue can be used where child processes read from queue, compute and send the data"""
receivedMatrixA = None
receivedMatrixB = None
totalProcessors = comm.Get_size()


for i in range(0, ARows):

    for j in range(0, BColumns):

        scatterRow = None
        scatterColumn = None


        if currentRank == root:

            scatterRow = np.array_split(matrixA[i, :], totalProcessors)
            scatterColumn = np.array_split(matrixB[:, j], totalProcessors)


        """No need to send the data again and again if there is only one row in matrixA or one column in matrixB 
        following check will take care of the respective cases"""


        if scatterMatrixA:
            receivedMatrixA = comm.scatter(scatterRow, root)

            """If there is only one row no need to scatter the data again and again"""
            if ARows == 1:
                scatterMatrixA = False

        if scatterMatrixB:
            receivedMatrixB = comm.scatter(scatterColumn, root)

            """If there is only one column no need to scatter data again and again"""
            if BColumns ==1:
                scatterMatrixB = False

        
        """Perform element wise product and add it before sending. 
        This will reduce the newtwork traffic/Less data shared among processes"""

        data = comm.gather(np.dot(receivedMatrixA, receivedMatrixB), root)
        
        if currentRank == root:
            matrixC[i][j] = np.sum(data)



if currentRank == root:
    print("Matrix A")
    print(matrixA)
    print("Matrix B")
    print(matrixB)
    print("Matrix C = MatrixA * MatrixB")
    print(matrixC)
    """COMMENT FOLLOWING LINE TO PREVENT STORAGE OF MATRIX TO DISK"""
    #np.savetxt("MatrixC", matrixC)


endTime = MPI.Wtime()

if currentRank != root:
    print("-->Time taken on processor "+ str(currentRank) + " - "+ str(endTime - globalStartTime))

else:
    print("-->Time taken on processor "+ str(currentRank) + " - "+ str((endTime - globalStartTime) - (dataGenerationEnded - dataGenerationStarted)))
    print("-->Total time taken including data generation "+ str(currentRank) + " - "+ str(endTime - globalStartTime))
