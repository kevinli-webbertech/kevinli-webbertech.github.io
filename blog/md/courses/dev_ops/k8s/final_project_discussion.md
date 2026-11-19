# Final Project Discussion

This solution deploys Jenkins LTS, Nexus, and MySQL on Minikube, then uses Jenkins to build a Spring Boot JAR, publish artifacts to Nexus, and update a Kubernetes deployment.

## Deployment flow

1. Check out source code from GitHub.
2. Run Maven tests and package the JAR.
3. Publish the JAR to a Nexus Maven hosted repository.
4. Build and push a container image to a Nexus Docker hosted registry.
5. Update the Kubernetes Deployment and wait for its rollout.

The `gs-spring-boot` Jenkinsfile runs checkout, test, package, and archive stages, then **publishes the JAR to Nexus**, **builds and pushes a container image**, and **deploys the image to Kubernetes** after artifact archiving.

## Jenkinsfile

Use the `gs-spring-boot` pipeline definition:

```text
https://github.com/kevinli-webbertech/gs-spring-boot/blob/main/Jenkinsfile
```

## Kubernetes files

All manifests and the deployment script are in [`../homework/scripts/`](../homework/scripts/):

* `jenkins-deployment.yaml` and `jenkins-service.yaml`
* `nexus-deployment.yaml` and `nexus-service.yaml`
* `mysql-configmap.yaml`, `mysql-secret.yaml`, `mysql-pvc.yaml`, `mysql-deployment.yaml`, and `mysql-service.yaml`
* `network-policy.yaml`
* `deploy.sh`

Run the deployment from that directory:

```bash
chmod +x deploy.sh
./deploy.sh
```

## Jenkins setup

1. Open Jenkins with `minikube service jenkins --url`.
2. Get the initial password:

   ```bash
   kubectl exec deployment/jenkins -- cat /var/jenkins_home/secrets/initialAdminPassword
   ```

3. Install suggested plugins and create the administrator account.
4. Add Jenkins credentials named `nexus-credentials` and `kubeconfig-credentials`.
5. Create a Pipeline job for `https://github.com/kevinli-webbertech/gs-spring-boot` and select its `Jenkinsfile`.
6. Run **Build Now** and capture screenshots of checkout, test, package, Nexus publishing, image push, and rollout.

## Nexus setup

Open Nexus with `minikube service nexus --url`. Retrieve its initial password with:

```bash
kubectl exec deployment/nexus -- cat /nexus-data/admin.password
```

Sign in as `admin`, change the password, then create Maven hosted and Docker hosted repositories. Replace all placeholder passwords before using this outside a local demonstration.

## Networking

Kubernetes provides in-cluster networking and Service DNS. Jenkins can reach Nexus at `http://nexus:8081`; the web application reaches MySQL at `mysql-service:3306`. `network-policy.yaml` explicitly permits Jenkins-to-Nexus traffic on TCP 8081 and Jenkins/webapp-to-MySQL traffic on TCP 3306 when the CNI supports NetworkPolicy enforcement.

## Testing

```bash
kubectl get pods,deployments,svc,pvc,networkpolicy
kubectl rollout status deployment/mysql
kubectl rollout status deployment/nexus
kubectl rollout status deployment/jenkins
kubectl logs deployment/jenkins
kubectl logs deployment/nexus
```

Expected result: the MySQL, Nexus, and Jenkins Pods are `Running` and `1/1` Ready; each deployment completes its rollout; `mysql-pvc` is `Bound`; and the three Services plus `allow-ci-and-database-traffic` NetworkPolicy exist.
