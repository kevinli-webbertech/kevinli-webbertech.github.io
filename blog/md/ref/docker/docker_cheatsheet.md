# Docker Cheatsheet

## Building image

`docker build --tag alpine-xvfb:beta1 .`

`docker build --no-cache --tag alpine-xvfb:beta1 .`

## Building with parameter

`docker build -t jdk21:lastest --build-arg DTR_URL=$DTR_URL .`

## Running without container deletion unpon exit

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
