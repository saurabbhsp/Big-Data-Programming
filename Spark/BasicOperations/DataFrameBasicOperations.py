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

df = session.read.json("/home/saurabh/Documents/git-repo/Bigdata/Big-Data-Programming/Spark/BasicOperations/Data/students.json")
df.show()

df = df.na.fill(df.groupBy().mean('points').collect()[0][0], "points")
df = df.na.fill("unknown", "dob")
df = df.na.fill("--", "last_name")
df.show()

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


df = df.select("course", "dob", calculateAge("dob").alias("age"),
               "first_name", "last_name", "points", "s_id")

df.show()
df_stats = df.select((stddev("points") + mean("points")).alias('one_std_dev'))
val = df_stats.collect()[0][0]


df = df.withColumn("points", when(df.points > val, 20).otherwise(df.points))
df.show()

hist = df.select('points').rdd.flatMap(lambda x: x).collect()
print(hist)
plt.hist(hist, bins = 20)
plt.show()
