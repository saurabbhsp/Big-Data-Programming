from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, when, isnull, sum, count, lit, avg, stddev
from pyspark.sql.window import Window

session = SparkSession.builder.appName("TaggingSession").getOrCreate()
sc = session.sparkContext
sc.setLogLevel("ERROR")

df = sc.textFile("/home/saurabh/Documents/git-repo/Bigdata/Big-Data-Programming/Spark/BasicOperations/Data/tags.dat").map(lambda x: x.split('::')).map(lambda x: [int(x[0]), int(x[1]), x[2], int(x[3])]).toDF()
df.show()

df = df.select(df._1.alias("UserId"), df._2.alias("MovieId"),
               df._3.alias("Tag"), df._4.alias("TimeStamp"))
df.show()

window_partition = Window.partitionBy('UserId').orderBy(['UserId', 'TimeStamp'])

df = df.withColumn("lagged", lag(df.TimeStamp).over(window_partition))
df.show()
df = df.withColumn("SessionTime", when(isnull(df.TimeStamp - df.lagged), 0).otherwise(df.TimeStamp - df.lagged))


df = df.withColumn("sessionTimeOut", when(df.SessionTime > (30 * 60), 1).otherwise(0))
df.show()

window_partition = Window.partitionBy("UserId").orderBy('TimeStamp')
df = df.withColumn("SessionId", sum(df.sessionTimeOut).over(window_partition))
df.orderBy('MovieId').show()

df = df.withColumn("SessionId", df.SessionId + lit(1))
df.show()
tagging_frequency = df.groupBy(['UserId', 'SessionId']).agg(count('SessionId').alias('Frequency'))
tagging_frequency.show()

stat_a = tagging_frequency.groupBy(['UserId']).agg(avg('Frequency').alias('Average'), stddev('Frequency').alias('StdDev'))
stat_a.show()

stat = tagging_frequency.groupBy().agg(avg('Frequency').alias('Average'), stddev('Frequency').alias('StdDev')).collect()
print(stat)
stat_a.filter(stat_a['Average'] > 2 * stat[0][0] + stat[0][1]).show()
