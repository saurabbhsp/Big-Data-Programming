## 1. Execution Environment
Following code has been executed with python 3.5.2 with MPI version 1.10.2<br/>
OS: Mint 18.3 sylvia<br/>
Kernel: x86_64 Linux 4.13.0-38-generic<br/>
CPU: Intel Core i7-7700HQ CPU @ 3.8GHz<br/>
RAM: 23973MiB


## 2. Common Functions

Following are the custom implementations of scatter and gather functions for collective communication using MPI_SEND and MPI_RECV.

### 2.1. Scatter

```python
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
 ```
The above method accepts data to be sent to all processors and root processor which is sending the data. The data is then  broken into chunks depending on the processor count.
### 2.2. Gather
```python
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
```
The above method gathers the data from all processors on root node. This is exact opposite of scatter function.
## 3. Add Vectors
This project generates two big vectors and adds them. These vectors are generated randomly using numpy.random.randint method. Another alternative to this is to read the vectors from file system.<br />
For running the add vectors snippet execute following shell script
```console
sh AddVectors.sh {no of processors} {vector length}
```

AddVectors.sh will execute AddVectors.py using mpiexec
```console
mpiexec -np $1 python3 AddVectors.py $2
```

Root processor will scatter two vectors on all processors. Each processor will add these two vectors in parallel and then will send the data back to the root using scatter.
```python
vecA = scatter(vectorA, root)
vecB = scatter(vectorB, root)
vecC = vecA + vecB
vectorC = gather(vecC, root)
```

#### 3.1 Performanace
| Vector Size        | Processors           | Execution Time (Including data generation time)|
| ------------- |:-------------:| -----:|
| 10^7      | 3 | 0.5744030475616455 |
| 10^7      | 6      |   0.5959308147430492 |
| 10^7 | 7      |    0.5316071510314941 |
<br/>
PS:- Coment block that stores the vector to disk to get accurate results.

## 4. Vector Average
This projects tries to calculate average of a big vector. Vector of given size is randomly created. Alternatively it can be read from a file.
<br />
For running the vector average snippet execute following shell script
```console
sh VectorAverage.sh {no of processors} {vector length}
```

VectorAverage.sh will execute VectorAverage.py using mpiexec
```console
mpiexec -np $1 python3 VectorAverage.py $2
```

Root processor will scatter vector on all processors. Each processor will add the numbers in scattered vector and send the result back to the root. Root will calculate the average of this big sum.
```python
vecSum = np.sum(scatter(vectorA, root))
assimilatedVector = gather(vecSum, root)

if currentRank == root:
	bigSum = np.sum(assimilatedVector)
	average = (bigSum*1.0)/int(eval(sys.argv[1]))
	print("The average of the vector is "+str(average))
```

#### 4.1 Performanace
| Vector Size        | Processors           | Execution Time (Including data generation time)|
| ------------- |:-------------:| -----:|
| 10^7      | 3 | 0.22130084037780762 |
| 10^7      | 6      |   0.22015881538391113 |
| 10^7 | 7      |    0.20660400390625000 |
<br/>
PS:- Coment block that stores the vector to disk to get accurate results.


## 5. Matrix multiplication
For parallel matrix multiplication there are many algorithms available. However the general idea is to eiher use multiple processors to compute different matrix values or salvage the processors to compute the same matrix values. Different algorithms like Cannon and Fox salvage varient of these ideas.<br />
I am using different processors to generate the same value. 
```python
"""For special case where the input is vector. No need to resend the data again"""
scatterMatrixA = True
scatterMatrixB = True


"""Matrix dimesions are needed on all systems. Not actual data but just dimensions
Alternatively a message queue can be used where child processes read from queue,
compute and send the data"""
receivedMatrixA = None
receivedMatrixB = None

for i in range(0, ARows):

    for j in range(0, BColumns):

        scatterRow = None
        scatterColumn = None


        if currentRank == root:

            scatterRow = matrixA[i, :]
            scatterColumn = matrixB[:, j]


        """No need to send the data again and again if there is only one row 
        in matrixA or one column in matrixB 
        following check will take care of the respective cases"""


        if scatterMatrixA:
            receivedMatrixA = scatter(scatterRow, root)

            """If there is only one row no need to scatter the data again and again"""
            if ARows == 1:
                scatterMatrixA = False

        if scatterMatrixB:
            receivedMatrixB = scatter(scatterColumn, root)

            """If there is only one column no need to scatter data again and again"""
            if BColumns ==1:
                scatterMatrixB = False

        
        """Perform element wise product and add it before sending. 
        This will reduce the newtwork traffic/Less data shared among processes"""

        data = gather(np.dot(receivedMatrixA, receivedMatrixB), root)
        
        if currentRank == root:
            matrixC[i][j] = np.sum(data)
```

The general algorithm above is to compute dot product of two vectors in parallel. Iterate over the rows of first matrix and columns of second matrix. Compute the resultant dot product.
```
for row in matrix A:
	for column in matrixB:
    	//Execute dot product in parallel by scattering and gathering the vectors
        result[row][column] = resultantDotProduct
```

Vector multiplication is a special case of vector multiplication, however you do not need to scatter the data again and again in case of a vector. This will essentially reduce the communication overhead and overhead of splitting the data again and again.In above code this special case is handled using scatterMatrixA and scatterMatrixB variables.

I have two implementations for matrix multiplication. One is using point to point communication by simulating scattering and gathering behaviour and other is using MPI collective communication scatter and gather methods.

To execute maxtrix multiplication call the following script from respective folders.
```console
sh MatrixMultiplication.sh {no of processors} {matrixA rows} {matrixA columns} {matrixB rows} {matrixB columns}
```

MatrixMultiplication.sh will execute MatrixMultiplication.py using mpiexec
```console
mpiexec -n $1 python3 MatrixMultiplication.py $2 $3 $4 $5
```


#### 5.1 Performanace
| Matrix A dimensions        | Matrix B dimensions        | Processors           | Execution Time (Including data generation time) point to point| Execution Time (Including data generation time) Collective communication|
| ------------- |:-------------:|:-------------:| -----:| -----:|
| 10^3 * 10^ 2      | 10^2 * 10^2 | 5|29.784949064254760 |32.935425996780396|
| 10^3 * 10^ 2      | 10^2 * 10^2 | 6| 24.615367889404297|26.97385001182556|
| 10^3 * 10^ 2      | 10^2 * 10^2 | 7| 40.534188985824585|43.2965669631958|

PS:- Coment block that stores the vector to disk to get accurate results.
