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

Run the following 

`kubectl apply -f https://k8s.io/examples/service/load-balancer-example.yaml`

or 

`kubectl apply -f load-balancer-example.yaml`

`kubectl get deployments hello-world`

## Ref

https://kubernetes.io/docs/tutorials/stateless-application/expose-external-ip-address/