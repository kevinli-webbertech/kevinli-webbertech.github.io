# Final Project Discussion

This solution deploys Jenkins LTS, Nexus, and MySQL on Minikube, then uses Jenkins to build a Spring Boot JAR, publish artifacts to Nexus, and update a Kubernetes deployment.

## Step 1: Get familiar with the gs-spring-boot project

[`gs-spring-boot`](https://github.com/kevinli-webbertech/gs-spring-boot/tree/jenkin_nexus_k8s_integration) is the reference project used throughout this discussion for the Jenkins → Nexus → Docker → Kubernetes pipeline. Your own Spring Boot webapp (with MySQL connectivity, per the final project tasks) should follow the same build and deploy pattern.

The repository does not currently have a `README.md`; the only root-level text file (`readme.txt`) is a placeholder, so there is no project README to link here yet.

Before wiring up Jenkins, build and run the JAR locally to confirm it works:

```bash
git clone --branch jenkin_nexus_k8s_integration https://github.com/kevinli-webbertech/gs-spring-boot.git
cd gs-spring-boot
./mvnw clean package -DskipTests
java -jar target/*.jar
```

With the app running, test it with a couple of API calls in another terminal:

```bash
curl http://localhost:8080/
# Expected: Greetings from Spring Boot!

curl http://localhost:8080/actuator/health
# Expected: {"status":"UP"}
```

If both calls return the expected output, the JAR is running correctly. This branch also adds a JDBC-backed `messages` API (`Message`, `MessageRepository`, `MessageController`) backed by H2 locally and MySQL in Kubernetes, so verify that the database code is actually working before moving on:

1. **Run the automated test.** `MessageControllerTest` drives a full create/read/update/delete cycle through `MockMvc` against the real `JdbcTemplate`/`MessageRepository` (nothing is mocked), clearing the `messages` table before each run:

   ```bash
   ./mvnw test
   ```

2. **Exercise the API locally (H2).** With the JAR from above still running:

   ```bash
   curl -X POST http://localhost:8080/api/messages \
     -H 'Content-Type: application/json' -d '{"content":"hello db"}'
   # 201 {"id":1,"content":"hello db","createdAt":"..."}

   curl http://localhost:8080/api/messages
   # 200 [{"id":1,"content":"hello db",...}]

   curl -X PUT http://localhost:8080/api/messages/1 \
     -H 'Content-Type: application/json' -d '{"content":"updated"}'
   # 200 {"id":1,"content":"updated",...}

   curl -i -X DELETE http://localhost:8080/api/messages/1
   # 204, then GET /api/messages/1 should 404
   ```

   A generated `id`/`createdAt` on POST, and the row reappearing on the following GET, confirms writes are really hitting the database rather than being held in memory by the controller.

3. **Exercise the API locally against real MySQL.** This is the closest local check to the `kubernetes` profile:

   ```bash
   mysql -u root -p -e "CREATE DATABASE spring_boot_db; CREATE USER 'springapp'@'%' IDENTIFIED BY 'replace-me'; GRANT ALL ON spring_boot_db.* TO 'springapp'@'%';"

   export SPRING_PROFILES_ACTIVE=kubernetes
   export SPRING_DATASOURCE_URL='jdbc:mysql://127.0.0.1:3306/spring_boot_db?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC'
   export SPRING_DATASOURCE_USERNAME=springapp
   export SPRING_DATASOURCE_PASSWORD=replace-me
   java -jar target/spring-boot-complete-0.0.1-SNAPSHOT.jar
   ```

   Repeat the `curl` calls from step 2, then confirm the row directly in MySQL, bypassing the app entirely:

   ```bash
   mysql -u springapp -p spring_boot_db -e "SELECT * FROM messages;"
   ```

### Deploy the JAR to Minikube manually

From the `gs-spring-boot` repository root, stop the local Java process, then package the JAR and build a Java 17 runtime image directly in Minikube. This first rollout uses the app's default H2 database.

```bash
minikube start
kubectl config use-context minikube
./mvnw clean package

cat > Dockerfile.manual <<'EOF'
FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/spring-boot-complete-0.0.1-SNAPSHOT.jar app.jar
ENTRYPOINT ["java", "-jar", "app.jar"]
EOF

minikube image build -t spring-boot-manual:1 -f Dockerfile.manual .
```

Apply the Deployment in the same namespace where you will deploy MySQL in Step 2:

```bash
kubectl apply -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
        - name: webapp
          image: spring-boot-manual:1
          imagePullPolicy: Never
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /actuator/health
              port: 8080
EOF

kubectl rollout status deployment/webapp --timeout=180s
kubectl port-forward deployment/webapp 8080:8080
```

In another terminal, run `curl -fsS http://localhost:8080/actuator/health` and check for `"status":"UP"`. Stop the port-forward with Ctrl+C before moving on.

## Step 2: Deploy MySQL and verify the application in Kubernetes

From the website repository root, change into the directory containing the five MySQL manifests listed under **Kubernetes files**. Edit the placeholder passwords in your local copy of `mysql-secret.yaml` before the first deployment. The ConfigMap supplies `myappdb` and `myappuser` by default. MySQL uses those settings and the Secret to initialize an empty volume. If `mysql-pvc` already contains a database, use the credentials and database configured in that instance.

```bash
cd blog/md/courses/dev_ops/homework/scripts
kubectl apply -f mysql-configmap.yaml -f mysql-secret.yaml -f mysql-pvc.yaml
kubectl apply -f mysql-service.yaml -f mysql-deployment.yaml
kubectl rollout status deployment/mysql --timeout=180s
kubectl get pvc mysql-pvc
kubectl get service mysql-service
```

Check that the claim is `Bound` and that MySQL accepts a TCP connection with the application credentials. If the query fails while MySQL is initializing, check `kubectl logs deployment/mysql` and retry the query before continuing.

```bash
kubectl exec deployment/mysql -- sh -c   'MYSQL_PWD="$MYSQL_PASSWORD" mysql --protocol=TCP -h 127.0.0.1 -u "$MYSQL_USER" "$MYSQL_DATABASE" -e "SELECT 1;"'
```

Switch the Step 1 Deployment to the `kubernetes` Spring profile. The Pod reads the database name and username from the ConfigMap and the password from the Secret. Keep `MYSQL_DATABASE` above `SPRING_DATASOURCE_URL` so Kubernetes can expand it in the JDBC URL.

```bash
kubectl patch deployment webapp --type=strategic --patch "$(cat <<'EOF'
spec:
  template:
    spec:
      containers:
        - name: webapp
          env:
            - name: MYSQL_DATABASE
              valueFrom:
                configMapKeyRef:
                  name: mysql-config
                  key: MYSQL_DATABASE
            - name: SPRING_PROFILES_ACTIVE
              value: kubernetes
            - name: SPRING_DATASOURCE_URL
              value: 'jdbc:mysql://mysql-service:3306/$(MYSQL_DATABASE)?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC'
            - name: SPRING_DATASOURCE_USERNAME
              valueFrom:
                configMapKeyRef:
                  name: mysql-config
                  key: MYSQL_USER
            - name: SPRING_DATASOURCE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mysql-secret
                  key: MYSQL_PASSWORD
EOF
)"
kubectl rollout status deployment/webapp --timeout=180s
kubectl port-forward deployment/webapp 8080:8080
```

Leave the port-forward running. In a second terminal, create a message through the application, then query the MySQL Pod directly:

```bash
curl -fsS http://localhost:8080/actuator/health
curl -fsS -X POST http://localhost:8080/api/messages   -H 'Content-Type: application/json' -d '{"content":"from k8s"}'

kubectl exec deployment/mysql -- sh -c   'MYSQL_PWD="$MYSQL_PASSWORD" mysql --protocol=TCP -h 127.0.0.1 -u "$MYSQL_USER" "$MYSQL_DATABASE" -e "SELECT id, content FROM messages ORDER BY id DESC LIMIT 10;"'
```

Match the ID and content from the POST response to the MySQL row. This confirms that the webapp wrote through `mysql-service` to the MySQL Pod. Capture the matching outputs for the final report.


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

All manifests and the deployment script are rendered as individual pages below:

* [`jenkins-deployment.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/jenkins-deployment-yaml.html) and [`jenkins-service.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/jenkins-service-yaml.html)
* [`nexus-deployment.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/nexus-deployment-yaml.html) and [`nexus-service.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/nexus-service-yaml.html)
* [`mysql-configmap.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/mysql-configmap-yaml.html), [`mysql-secret.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/mysql-secret-yaml.html), [`mysql-pvc.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/mysql-pvc-yaml.html), [`mysql-deployment.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/mysql-deployment-yaml.html), and [`mysql-service.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/mysql-service-yaml.html)
* [`network-policy.yaml`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/network-policy-yaml.html)
* [`deploy.sh`](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/scripts/deploy-sh.html)

The source files live together in [`blog/md/courses/dev_ops/homework/scripts/`](https://github.com/kevinli-webbertech/kevinli-webbertech.github.io/tree/main/blog/md/courses/dev_ops/homework/scripts) in the repository. Download them into one directory and run the deployment from there:

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
