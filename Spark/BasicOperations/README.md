# Spark basic operations
Following code deals with spark data manipulation. Following code is written python and uses pyspark with python 3.5.2 as python backend.

## Creating local clusters
For executing the a local cluster is created. All the code is submitted to the master using spark submit interface.

## 1 Basic operations on RDD
### 1.1 List to RDD
Given two RDD the following code will load the data from the list to the spark RDD
The data is mapped as key value pair and the value is stored as key and index in the list is stored as value. This has been done using parallelize command.
 ```python
 from pyspark.sql import SparkSession

session = SparkSession.builder.appName("RDDBasics").getOrCreate()
sc = session.sparkContext
sc.setLogLevel("ERROR")

"""Create RDD"""
a = ["spark", "rdd", "python",
     "context", "create", "class"]
b = ["operation", "apache", "scala", "lambda",
     "parallel", "partition"]
rdd_a = sc.parallelize((value, key) for key, value in enumerate(a))
rdd_b = sc.parallelize((value, key) for key, value in enumerate(b))
```
### 1.2 Spark RDD joins
Follwing code will run right and full outer join on the created lists.
```python
"""right outer join"""
"""Collect should not be used on real data set."""
result = rdd_a.rightOuterJoin(rdd_b)
print(result.collect())
"""full outer join"""
result = rdd_a.fullOuterJoin(rdd_b)
print(result.collect())
```
```console
[('parallel', (None, 4)), ('lambda', (None, 3)), ('scala', (None, 2)),
 ('operation', (None, 0)), ('apache', (None, 1)), ('partition', (None, 5))]


[('python', (2, None)), ('spark', (0, None)), ('create', (4, None)),
('context', (3, None)), ('parallel', (None, 4)), ('lambda', (None, 3)),
('class', (5, None)), ('rdd', (1, None)), ('scala', (None, 2)),
('operation', (None, 0)), ('apache', (None, 1)), ('partition', (None, 5))]

```
The first output is from right outer join. The right outer join will take all values from the right RDD. In above case all data from rdd_b is provided as output. The output is a tuple with key as key of rdd_b and value is a tuple as (rdd_a_value, rdd_b_value). Places where the key of rdd_b is not present in rdd_a the value is None.

Second output is from full outer join. Similar to right outer join but now the values of both rdd_a and rdd_b keys are included.

### 1.3 Map reduce
Following code uses map reduce to calculate occurance of character 's'
```python
"""map-reduce"""
a_count = rdd_a.map(lambda pair: pair[0].lower().count('s')).reduce(lambda x,
                                                                    y: x+y)
b_count = rdd_b.map(lambda pair: pair[0].lower().count('s')).reduce(lambda x,
                                                                    y: x+y)

print("The total count for character s in both RDD using map reduce is is " +
      str(a_count + b_count))
```
### 1.4 Aggregate function
Following code does the same instead of map reduce it uses aggregate functions.

```python
a_count = rdd_a.aggregate((0, 0), lambda x, y: (1, y[0].lower().count('s') +
                                                x[1]), lambda x, y: (1, y[1] +
                                                                     x[1]))

b_count = rdd_b.aggregate((0, 0), lambda x, y: (1, y[0].lower().count('s') +
                                                x[1]), lambda x, y: (1, y[1] +
                                                                     x[1]))
print("The total count for character s in both RDD using aggregate" +
      " function is " + str(a_count[1] + b_count[1]))

 ```
![Output](https://drive.google.com/uc?export=view&id=1sgVywDCj2v0Q3-YiOfaekSjQ5pIVcu2u)

## 2 Dataframe basic operations

### 2.1 Load json
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import date_format, coalesce, to_date,\
                                  current_timestamp, datediff, stddev, mean,\
                                  when

import matplotlib.pyplot as plt

session = SparkSession.builder.appName("DataFrameBasics").getOrCreate()
sc = session.sparkContext
sc.setLogLevel("ERROR")

"""Using custom schema so that points can
be loaded as double"""

df = session.read.json("/home/saurabh/python-spark/Spark/" +
                       "Getting_Started/Data/students.json")
df.show()
```
The above code will read json file and load it as spark dataframe. Following is the output of the above code.

![Output](https://drive.google.com/uc?export=view&id=1paB2vIErk_cxUFawQwEWGcr17H0Tpt3p)


The data contains some null values. The next step is to remove null values removal.

### 2.2 Null value replacement

```python
df = df.na.fill(df.groupBy().mean('points').collect()[0][0], "points")
df = df.na.fill("unknown", "dob")
df = df.na.fill("--", "last_name")
df.show()
```
The above code results into following output
![Output](https://drive.google.com/uc?export=view&id=1jfrBZuYJbJr3HZbHiDKG8C-o6gUFOrmk)

### 2.3 Date manipulation
#### 2.3.1 Date Formatting

```python
"""user defined functions"""


def parseDate(col, formats=("MMM dd, yyyy", "dd MMM yyyy")):
    """coalesce - Retun first non null argument
    * converts the list to parameters"""
    return coalesce(*[to_date(col, f) for f in formats])


def calculateAge(col):
    return datediff(current_timestamp(), to_date(col, "dd-MM-yyyy")) / 365


df = df.select("course", date_format(parseDate("dob"),
                                     "dd-MM-yyyy").alias("dob"),
               "first_name", "last_name", "points", "s_id")
df.show()
```
![Output](https://drive.google.com/uc?export=view&id=1xXKhP25eVMb4ETWs9MIE14eGF6DN_-6P)

#### 2.3.2 Age calculation
```python

df = df.select("course", "dob", calculateAge("dob").alias("age"),
               "first_name", "last_name", "points", "s_id")

df.show()
```
![Output](https://drive.google.com/uc?export=view&id=1X7dXHAO89LAYWER2pXXWMiqHoodkf_-N)

### 2.4 Stastical functions

```python

df_stats = df.select((stddev("points") + mean("points")).alias('one_std_dev'))
val = df_stats.collect()[0][0]

df = df.withColumn("points", when(df.points > val, 20).otherwise(df.points))
df.show()
```

```python
hist = df.select('points').rdd.flatMap(lambda x: x).collect()
print(hist)
plt.hist(hist, bins = 20)
plt.show()

```
![Output](https://drive.google.com/uc?export=view&id=19LCVoGaRHZlberx7pPEoI1gt6ZQbwGWw)

#### Distribution of score

![Output](https://drive.google.com/uc?export=view&id=1XDd29uJgzpm3Bq5EYNxLegrjgxMND6vF)

## 3 Recommender system dataset
For this section Movie lens data set is used.
## 3.1 Load Data in Dataframe
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, when, isnull, sum, count, lit, avg, stddev
from pyspark.sql.window import Window

session = SparkSession.builder.appName("TaggingSession").getOrCreate()
sc = session.sparkContext
sc.setLogLevel("ERROR")

df = sc.textFile("/home/saurabh/Documents/git-repo/Bigdata/Big-Data-Programming/"+
"Spark/BasicOperations/Data/tags.dat").map(
lambda x: x.split('::')).map(
lambda x: [int(x[0]), int(x[1]), x[2], int(x[3])]).toDF()

df.show()

df = df.select(df._1.alias("UserId"), df._2.alias("MovieId"),
               df._3.alias("Tag"), df._4.alias("TimeStamp"))
df.show()
```
![Output](https://drive.google.com/uc?export=view&id=12OYaUUWNW3j3-J2phH1UnybNmDtXSXfI)

### 3.2 Tag session
Following code will tag user for session. If user remains inactive for 30 minutes. It is considered as new session.
```python
window_partition = Window.partitionBy('UserId').orderBy(['UserId', 'TimeStamp'])

df = df.withColumn("lagged", lag(df.TimeStamp).over(window_partition))
df.show()
df = df.withColumn("SessionTime",
  when(isnull(df.TimeStamp - df.lagged), 0).otherwise(df.TimeStamp - df.lagged))


df = df.withColumn("sessionTimeOut",
      when(df.SessionTime > (30 * 60), 1).otherwise(0))
df.show()

window_partition = Window.partitionBy("UserId").orderBy('TimeStamp')
df = df.withColumn("SessionId", sum(df.sessionTimeOut).over(window_partition))
df.orderBy('MovieId').show()

df = df.withColumn("SessionId", df.SessionId + lit(1))
df.show()
```
![Output](https://drive.google.com/uc?export=view&id=1zlxey4hOoUOeZalpZwzyIsU37udAbwsr)

### 3.3 Session Stats

#### 3.3.1 Calculate Frequency
```python
tagging_frequency = df.groupBy(['UserId',
                      'SessionId']).agg(count('SessionId').alias('Frequency'))
tagging_frequency.show()
```
![Output](https://drive.google.com/uc?export=view&id=1-EjGGMhZ9P7BHD6LODe2oBIcxFETxksW)

#### 3.3.2 Avg and Std user frequency for each user
```python
stat = tagging_frequency.groupBy(['UserId']
            ).agg(avg('Frequency').alias('Average'),
             stddev('Frequency').alias('StdDev'))
stat.show()
```
![Output](https://drive.google.com/uc?export=view&id=1Z-F4gJmSeBDi2VT30SlyyV1CANaUXkeq)

#### 3.3.3 Avg and Std user frequency across users
```python
stat = tagging_frequency.groupBy().agg(
      avg('Frequency').alias('Average'),
       stddev('Frequency').alias('StdDev')).collect()
print(stat)
```
![Output](https://drive.google.com/uc?export=view&id=1LWo-dOwAfog1YBAc-aJPZ6p2KTMX18Eg)

#### 3.3.4 Users with mean more that three standard deviation

```python
stat_a.filter(stat_a['Average'] > 2 * stat[0][0] + stat[0][1]).show()
```
![Output](https://drive.google.com/uc?export=view&id=1NCpxXgnKMeFuF6tS_FLUZGFETmzg8jjj)
