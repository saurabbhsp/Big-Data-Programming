from pyspark.sql import SparkSession

session = SparkSession.builder.appName("RDDBasics").getOrCreate()
sc = session.sparkContext
sc.setLogLevel("ERROR")

"""Create RDD"""
a = ["spark", "rdd", "python",
     "context", "create", "class"]
b = ["operation", "apache", "scala", "lambda",
     "parallel", "partition"]
# rdd_a = sc.parallelize((value, 1) for key, value in enumerate(a))
# rdd_b = sc.parallelize((value, 1) for key, value in enumerate(b))
rdd_a = sc.parallelize(a)
rdd_b = sc.parallelize(b)

"""right outer join"""
"""Collect should not be used on real data set."""
result = rdd_a.rightOuterJoin(rdd_b).map(lambda x: (x[0]+x[1][0], x[0]+x[1][1]))
print(result.collect())
"""full outer join"""
result = rdd_a.fullOuterJoin(rdd_b)
print(result.collect())


"""map-reduce"""
a_count = rdd_a.map(lambda pair: pair[0].lower().count('s')).reduce(lambda x,
                                                                    y: x+y)
b_count = rdd_b.map(lambda pair: pair[0].lower().count('s')).reduce(lambda x,
                                                                    y: x+y)

print("The total count for character s in both RDD using map reduce is is " +
      str(a_count + b_count))

a_count = rdd_a.aggregate((0, 0), lambda x, y: (1, y[0].lower().count('s') +
                                                x[1]), lambda x, y: (1, y[1] +
                                                                     x[1]))

b_count = rdd_b.aggregate((0, 0), lambda x, y: (1, y[0].lower().count('s') +
                                                x[1]), lambda x, y: (1, y[1] +
                                                                     x[1]))
print("The total count for character s in both RDD using aggregate" +
      " function is " + str(a_count[1] + b_count[1]))
