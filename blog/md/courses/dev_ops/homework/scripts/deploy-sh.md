# deploy.sh

Deployment script that starts Minikube, applies every manifest in the correct order, waits for each rollout, and prints the resulting cluster objects.

```bash
#!/usr/bin/env bash
set -euo pipefail
minikube start
kubectl apply -f mysql-configmap.yaml -f mysql-secret.yaml -f mysql-pvc.yaml
kubectl apply -f mysql-deployment.yaml -f mysql-service.yaml
kubectl apply -f nexus-deployment.yaml -f nexus-service.yaml
kubectl apply -f jenkins-deployment.yaml -f jenkins-service.yaml
kubectl apply -f network-policy.yaml
kubectl rollout status deployment/mysql
kubectl rollout status deployment/nexus
kubectl rollout status deployment/jenkins
kubectl get configmap,secret,pvc,deployment,pods,svc
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/k8s/final_project_discussion.html)
