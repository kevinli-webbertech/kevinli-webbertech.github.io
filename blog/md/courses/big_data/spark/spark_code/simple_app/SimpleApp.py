"""SimpleApp.py"""
from pyspark.sql import SparkSession

#logFile = "/opt/spark/work-dir/README.md"  # Should be some file on your system
logFile = "./README.md"
spark = SparkSession.builder.appName("SimpleApp").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
logData = spark.read.text(logFile).cache()

numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

spark.stop()
