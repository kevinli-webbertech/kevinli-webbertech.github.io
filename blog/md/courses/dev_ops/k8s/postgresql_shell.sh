#!/bin/bash

# K8s Local setup script

minikube stop
minikube start

kubectl apply -f postgres-configmap.yaml
kubectl apply -f psql-pv.yaml
kubectl apply -f psql-pv.yaml
kubectl apply -f psql-claim.yaml
kubectl apply -f ps-deployment.yaml
kubectl apply -f ps-service.yaml

echo "checking PV..."
kubectl get pv

echo "checking PVC..."
kubectl get pvc

echo "checking deployments..."
kubectl get deployments

echo "checking services..."
kubectl get svc

echo "checking pods..."
kubectl get pods

echo "logging to the database..."

pod_name=$(kubectl get pods|grep postgres|awk {'print $1'})

echo "Checking pod name..."
echo $pod_name

status=$(get pods $pod_name --no-headers -o custom-columns=":status.phase"|awk {'print $1'})

while [ $status != 'Running' ]
  echo "pod is starting..."
  sleep 1
done 

echo "pod is ready to use..."

# kubectl exec -it $pod_name -- psql -h localhost -U ps_user --password -p 5432 ps_db
