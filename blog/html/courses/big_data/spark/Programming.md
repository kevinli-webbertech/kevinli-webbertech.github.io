# Programming

```shell
xiaofengli@xiaofenglx:~$ docker pull apache/spark:3.5.1
3.5.1: Pulling from apache/spark
3c67549075b6: Pull complete
482e8906db89: Pull complete
486b105cee21: Pull complete
ba536847d6fa: Pull complete
ef7346692452: Pull complete
318ff7bb2b26: Pull complete
8333604c87bb: Pull complete
18838527ab39: Pull complete
ec020c8178d4: Pull complete
4f4fb700ef54: Pull complete
3a9e183a5640: Pull complete
Digest: sha256:b49e3b73ce385c1693cc3294ec4b5b3882e9fa9bd5fc599d4c79198abc49cc94
Status: Downloaded newer image for apache/spark:3.5.1
docker.io/apache/spark:3.5.1
xiaofengli@xiaofenglx:~$ docker run -it apache/spark:3.5.1 /bin/bash
spark@bbe3927c7ff0:/opt/spark/work-dir$ 
```

Then we would go up one dir, and run the pyspark,

```shell
spark@bbe3927c7ff0:/opt/spark/work-dir$ cd ..
spark@bbe3927c7ff0:/opt/spark$ ls
bin  data  examples  jars  python  R  RELEASE  sbin  tests  work-dir
spark@bbe3927c7ff0:/opt/spark$ ./bin/pyspark
Python 3.8.10 (default, Nov 22 2023, 10:22:35)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
24/10/15 00:14:08 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Welcome to
____              __
/ __/__  ___ _____/ /__
_\ \/ _ \/ _ `/ __/  '_/
/__ / .__/\_,_/_/ /_/\_\   version 3.5.1
/_/

Using Python version 3.8.10 (default, Nov 22 2023 10:22:35)
Spark context Web UI available at http://bbe3927c7ff0:4040
Spark context available as 'sc' (master = local[*], app id = local-1728951249167).
SparkSession available as 'spark'.
>>> 
```

But before we write any code, let us dive out and see what is in the container,

```shell
...
Welcome to
____              __
/ __/__  ___ _____/ /__
_\ \/ _ \/ _ `/ __/  '_/
/__ / .__/\_,_/_/ /_/\_\   version 3.5.1
/_/

Using Python version 3.8.10 (default, Nov 22 2023 10:22:35)
Spark context Web UI available at http://bbe3927c7ff0:4040
Spark context available as 'sc' (master = local[*], app id = local-1728951249167).
SparkSession available as 'spark'.
Use quit() or Ctrl-D (i.e. EOF) to exit
>>> quit()
```

* Check the release and distribution of the docker image,

```shell
spark@bbe3927c7ff0:/opt/spark$ cat /etc/*release*
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=20.04
DISTRIB_CODENAME=focal
DISTRIB_DESCRIPTION="Ubuntu 20.04.6 LTS"
NAME="Ubuntu"
VERSION="20.04.6 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.6 LTS"
VERSION_ID="20.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=focal
UBUNTU_CODENAME=focal
```

* Check spark related vars

```shell
spark@bbe3927c7ff0:/opt/spark$ set |grep SPARK
SPARK_HOME=/opt/spark
SPARK_TGZ_ASC_URL=https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz.asc
SPARK_TGZ_URL=https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
```