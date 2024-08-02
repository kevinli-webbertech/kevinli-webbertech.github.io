# Lab 2 Helloworld Example using Load Balancer

## Create a load-balancer-example.yaml

service/load-balancer-example.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app.kubernetes.io/name: load-balancer-example
  name: hello-world
spec:
  replicas: 5
  selector:
    matchLabels:
      app.kubernetes.io/name: load-balancer-example
  template:
    metadata:
      labels:
        app.kubernetes.io/name: load-balancer-example
    spec:
      containers:
      - image: gcr.io/google-samples/hello-app:2.0
        name: hello-world
        ports:
        - containerPort: 8080
```

The ReplicaSet has five Pods each of which runs the Hello World application.

**Run minikube server**

![minikube_start](https://kevinli-webbertech.github.io/blog/images/k8s/minikube_start.png)

**Run the following to create deployment**

`kubectl apply -f https://k8s.io/examples/service/load-balancer-example.yaml`

or,

`kubectl apply -f load-balancer-example.yaml`

![create_deployment](https://kevinli-webbertech.github.io/blog/images/k8s/create_deployment.png)

**Display information about the Deployment**

`kubectl get deployments hello-world`

`kubectl describe deployments hello-world`

Output,

![deployments](https://kevinli-webbertech.github.io/blog/images/k8s/deployments.png)

A more detailed description of the `describe` sub command.

![deployment_description](https://kevinli-webbertech.github.io/blog/images/k8s/deployment_description.png)

**Display information about your ReplicaSet objects**

`kubectl get replicasets`

`kubectl describe replicasets`

Output of the above commands,

![ReplicaSet](https://kevinli-webbertech.github.io/blog/images/k8s/ReplicaSet.png)

**Create a Service object that exposes the deployment**

`kubectl expose deployment hello-world --type=LoadBalancer --name=my-service`

**Display information about the Service**

`kubectl get services my-service`

![k8s_service](https://kevinli-webbertech.github.io/blog/images/k8s/k8s_service.png)

**Display pods information**

`kubectl get pods --output=wide`

![k8s_pod](https://kevinli-webbertech.github.io/blog/images/k8s/k8s_pods.png)

**Test Web Application**

`curl http://<external-ip>:<port>`

where <external-ip> is the external IP address (LoadBalancer Ingress) of your Service, and <port> is the value of Port in your Service description. If you are using minikube, typing minikube service my-service will automatically open the Hello World application in a browser.

The response to a successful request is a hello message:

```shell
Hello, world!
Version: 2.0.0
Hostname: 0bd46b45f32f
```

![launch_app](https://kevinli-webbertech.github.io/blog/images/k8s/launch_app.png)

**Clean up**

`kubectl delete services my-service`

`kubectl delete deployment hello-world`

## Ref

- https://kubernetes.io/docs/tutorials/stateless-application/expose-external-ip-address/