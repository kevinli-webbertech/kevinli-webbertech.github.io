# Learning to write docker file

## Reviews

What we have learned previously, were the following topics,

* `docker pull` (download a docker image from docker.io)
* `docker run -d ...`, this is a command to run a docker image, if the image was not found in your local computer, it will go to do a `docker pull`.
* In the previous `Python App Deployment with Docker` we learn how to write a `flask` app in python and it is a simple web service deployed in docker.
In this following url,

https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/docker/python_app_deployment.html

We created a small python app in a few lines and built a docker image. And we run the docker image and we were able to see the following two goals,

* the website running
* using the `curl` command to query that the site is working and no 404 page.

## Today's goal

We will retrospect and inspect the docker file we provided earlier and see how that works.

