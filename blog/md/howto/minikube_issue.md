# Minikube start issue in Mac M Chip

Use home brew to install minikube and colima

```
$ minikube stop
$ minikube delete
$ rm -rf ~/.minikube/
$ colima start --arch x86_64 --memory 4
$ minikube start
```