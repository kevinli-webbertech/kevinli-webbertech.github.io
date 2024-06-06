# Minikube start issue in Mac M Chip

* Use home brew to install minikube and colima

* Make sure your docker desktop is closed entirely.

* Run `minikube start` if you run into any errors, then do the following.

```
$ minikube stop
$ minikube delete
$ rm -rf ~/.minikube/
$ colima start --arch x86_64 --memory 4
$ minikube start
```