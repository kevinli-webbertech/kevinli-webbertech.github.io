# Final Kubernetes Deployment

Please include the original questions in your homework report. Please check the syllabus for details or you will lose points.

K8s here we can use `minikube`.

## Tasks

Write a springboot helloworld project and that could connect to a mysql database and commit to github. (hint: please go to checkout in spring.io)

For **every task**, include at least one clearly labeled screenshot in the report that shows both the relevant configuration or command and successful proof of completion (for example, `kubectl get`, pod status, service output, pipeline output, or the running application). Screenshots must be legible and tied to the task number.

* Write a k8s deployment for Jenkins server. Include screenshots of the deployment manifest and Jenkins running. (10 pts)
* Write a k8s deployment for Nexus server. Include screenshots of the deployment manifest and Nexus running. (10 pts)
* Write a k8s deployment for MySQL database. Include screenshots of the deployment manifest and the database pod running. (10 pts)
* Write a k8s ConfigMap for MySQL username and password. Include screenshots of the ConfigMap YAML and `kubectl get configmap` output. (10 pts)
* Write a Jenkinsfile to check out code, build a JAR, and push it to the Nexus server. You may use a public GitHub repository for this task; this is acceptable. Provide the GitHub repository URL in your report. Include screenshots of the Jenkinsfile, successful pipeline stages, generated JAR, and Nexus upload. (10 pts)
* Write a k8s webapp deployment file to spin up the pod that runs the webapp. Include screenshots of the manifest, running pod, and a successful webapp response. (10 pts)
* Write a k8s PV deployment file. Include screenshots of the PV YAML and a bound or available PV. (10 pts)
* Write a k8s PVC deployment file. Include screenshots of the PVC YAML and a bound PVC. (10 pts)
* Write a service file. Include screenshots of the Service YAML and the service endpoint or successful application access. (10 pts)
* Write a shell script that checks all ConfigMaps, deployments, and services, starts the Minikube cluster, and deploys everything automatically. Include screenshots of the script and successful script output. (10 pts)

## Submission requirements

Submit one Word report (`.docx`) that includes the original questions, a short explanation for each task, and the required screenshots. The report must include a clickable link to the student's public GitHub repository that contains the Java/Spring Boot project and the `Jenkinsfile` used for this final project. Also submit the complete, runnable project source as a single ZIP file. The ZIP must include all required Kubernetes YAML files, the `Jenkinsfile`, and the startup/deployment shell script. Name files clearly so they can be matched to the tasks above:

* `jenkins-deployment.yaml`
* `nexus-deployment.yaml`
* `mysql-deployment.yaml`
* `mysql-configmap.yaml`
* `webapp-deployment.yaml`
* `pv.yaml`
* `pvc.yaml`
* `service.yaml`
* `Jenkinsfile`
* `deploy.sh`

Prefer a separate YAML file for each Kubernetes task. For every YAML file, document the exact `kubectl` command used to apply it and the command(s) used to verify it (for example, `kubectl get`, `kubectl describe`, or `kubectl logs`). Include these commands in the README and, where relevant, in the Word report.

Do not submit screenshots by themselves; embed them in the Word report. Include a README with any required setup steps, image names, URLs, and commands needed to run the project.
