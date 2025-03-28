Apache Spark
Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Scala, Java, Python, and R, and an optimized engine that supports general computation graphs for data analysis. It also supports a rich set of higher-level tools including Spark SQL for SQL and DataFrames, pandas API on Spark for pandas workloads, MLlib for machine learning, GraphX for graph processing, and Structured Streaming for stream processing.

https://spark.apache.org/

GitHub Actions Build PySpark Coverage PyPI Downloads

Online Documentation
You can find the latest Spark documentation, including a programming guide, on the project web page. This README file only contains basic setup instructions.

Building Spark
Spark is built using Apache Maven. To build Spark and its example programs, run:

./build/mvn -DskipTests clean package
(You do not need to do this if you downloaded a pre-built package.)

More detailed documentation is available from the project site, at "Building Spark".

For general development tips, including info on developing Spark using an IDE, see "Useful Developer Tools".

Interactive Scala Shell
The easiest way to start using Spark is through the Scala shell:

./bin/spark-shell
Try the following command, which should return 1,000,000,000:

scala> spark.range(1000 * 1000 * 1000).count()
Interactive Python Shell
Alternatively, if you prefer Python, you can use the Python shell:

./bin/pyspark
And run the following command, which should also return 1,000,000,000:

>>> spark.range(1000 * 1000 * 1000).count()
Example Programs
Spark also comes with several sample programs in the examples directory. To run one of them, use ./bin/run-example <class> [params]. For example:

./bin/run-example SparkPi
will run the Pi example locally.

You can set the MASTER environment variable when running examples to submit examples to a cluster. This can be spark:// URL, "yarn" to run on YARN, and "local" to run locally with one thread, or "local[N]" to run locally with N threads. You can also use an abbreviated class name if the class is in the examples package. For instance:

MASTER=spark://host:7077 ./bin/run-example SparkPi
Many of the example programs print usage help if no params are given.

Running Tests
Testing first requires building Spark. Once Spark is built, tests can be run using:

./dev/run-tests
Please see the guidance on how to run tests for a module, or individual tests.

There is also a Kubernetes integration test, see resource-managers/kubernetes/integration-tests/README.md

A Note About Hadoop Versions
Spark uses the Hadoop core library to talk to HDFS and other Hadoop-supported storage systems. Because the protocols have changed in different versions of Hadoop, you must build Spark against the same version that your cluster runs.

Please refer to the build documentation at "Specifying the Hadoop Version and Enabling YARN" for detailed guidance on building for a particular distribution of Hadoop, including building for particular Hive and Hive Thriftserver distributions.

Configuration
Please refer to the Configuration Guide in the online documentation for an overview on how to configure Spark.

Contributing
Please review the Contribution to Spark guide for information on how to get started contributing to the project.