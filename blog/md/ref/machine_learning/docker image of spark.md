# Getting a Docker Image of Apache Spark

## 1. Install Docker
If you haven't installed Docker yet, download and install it from the official Docker website: [Docker Installation Guide]((https://docs.docker.com/desktop/install/windows-install/)).

## 2. Pull the Official Spark Image
You can pull the official Spark image by running the following command in your terminal:
```bash
 docker pull bitnami/spark
```
This command pulls the Spark image maintained by **Bitnami**, which is a well-known provider of up-to-date and secure images.


## 3. Verify the Image
Once the image is pulled, you can verify it by listing the Docker images:

```bash
docker images
```
4. Run a Spark Container
   ```bash
   docker run -it --rm bitnami/spark spark-shell
```
This will start a Spark shell inside the Docker container.
