# Docker Cheatsheet

## Building image

`docker build --tag alpine-xvfb:beta1 .`

`docker build --no-cache --tag alpine-xvfb:beta1 .`

## Building with [Multiple]parameter[s]

`docker build -t jdk21:lastest --build-arg DTR_URL=$DTR_URL .`

A complete example,

To use it in another docker file, write a docker file,

```bash
ARG DCK_URL
ARG IMG_DIR
FROM ${Docker_URL}/${Image_DIR}/python3:3.0.7
ENTRYPOINT ["/bin/bash"]
```

Build it with multiple params, you will need multiple `--build-arg`,

```bash
docker build -t python3:ml 
    --build-arg DCK_URL=$DCR_URL
    --build-arg IMG_DIR=$IMG_DIR
.
```

Run and test it,

```bash
xiaofengli@xiaofenglx:~/code/docker_image/ml$ docker run -it python3:ml
[pythonuser@2507a4a1f071 ~]$ python --version
Python 3.11.7
```

## Running with container deletion unpon exit

`docker run --rm -it alpine-xvfb:beta1`
	
## Running with inline entrypoint

`docker run --entrypoint bash jdk21:latest -c "ls"`

## Getting into docker container

`docker exec -it <container_name> bash`

## Run an image and mount local drive

`docker run -v /tmp/test:/opt/test --rm -it alpine-xvfb:beta1`

`docker run -v $PWD:/opt/test --rm -it alpine-xvfb:beta1`

## Delete a particular image

`docker image rm $(docker image ls |grep xvfb| awk '{print $3}')`

## CMD[] vs Entrypoint[]

* CMD spawn off new process

* Entrypoint uses the same process
