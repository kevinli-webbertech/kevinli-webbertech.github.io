# Springboot deployment with Kubernetes

# Docker Container

A Docker container is a lightweight, standalone, executable package that includes everything needed to run a piece of software, including the code, runtime, libraries, dependencies, and configuration files. Docker containers are based on Docker technology, which allows developers to package their applications and all of their dependencies into a single container that can run consistently across different environments.

Key features of Docker containers include:

Isolation: Docker containers provide process and filesystem isolation, allowing applications to run in a sandboxed environment without interfering with other applications running on the same host.

Portability: Docker containers are portable and can run on any system that supports the Docker runtime, including laptops, servers, virtual machines, and cloud platforms. This portability makes it easy to develop, test, and deploy applications across different environments.

Efficiency: Docker containers are lightweight and use OS-level virtualization to share the host operating system's kernel, which reduces overhead and improves performance compared to traditional virtual machines.

Consistency: Docker containers encapsulate all of the dependencies and configuration required to run an application, ensuring consistent behavior across different environments and reducing the likelihood of "it works on my machine" issues.

Scalability: Docker containers are designed to be scalable and can be easily deployed and managed using container orchestration platforms like Kubernetes. This allows applications to scale up or down based on demand and ensures high availability and reliability.

Overall, Docker containers have revolutionized the way applications are packaged, distributed, and deployed, making it easier for developers to build, ship, and run software in a consistent and efficient manner.

## What is Kubernetes

Kubernetes is an open-source platform designed to automate deploying, scaling, and operating application containers. Originally developed by Google and now maintained by the Cloud Native Computing Foundation (CNCF), Kubernetes aims to provide a platform for automating the deployment, scaling, and operations of application containers across clusters of hosts. It provides a container-centric management environment and orchestrates computing, networking, and storage infrastructure on behalf of user workloads.

Kubernetes abstracts away the underlying infrastructure, allowing developers to deploy applications without worrying about the specific hardware or cloud provider details. It helps manage containerized applications by providing mechanisms for deployment, scaling, and resource management, ensuring that applications run consistently and reliably across different environments.

Key features of Kubernetes include:

1. **Container Orchestration**: Kubernetes automates the deployment, scaling, and management of containerized applications.

2. **Service Discovery and Load Balancing**: Kubernetes provides built-in mechanisms for service discovery and load balancing, allowing containers to communicate with each other and distributing traffic across multiple instances of an application.

3. **Self-healing**: Kubernetes automatically restarts containers that fail, replaces containers that become unresponsive, and kills containers that don't respond to user-defined health checks.

4. **Scaling**: Kubernetes can automatically scale the number of containers running an application based on CPU usage, memory consumption, or custom metrics.

5. **Rolling updates and Rollbacks**: Kubernetes supports rolling updates, allowing new versions of applications to be deployed with minimal downtime, and provides the ability to rollback to a previous version if needed.

6. **Resource Management**: Kubernetes allows users to specify the CPU and memory requirements for containers, ensuring that applications have access to the necessary resources while maximizing resource utilization.

7. **Storage Orchestration**: Kubernetes provides support for different types of storage solutions and allows users to mount storage volumes to containers.

Overall, Kubernetes has become the de facto standard for container orchestration in the cloud-native ecosystem, enabling organizations to build, deploy, and manage modern applications at scale.

## Installation of Minicube

* Download

`curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64`

* Installation

`sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64`

![start minikube](https://kevinli-webbertech.github.io/blog/images/springboot/minikube.png)


## Ref

- https://spring.io/guides/gs/spring-boot-kubernetes
- https://github.com/kubernetes/minikube
- https://minikube.sigs.k8s.io/docs/start/

