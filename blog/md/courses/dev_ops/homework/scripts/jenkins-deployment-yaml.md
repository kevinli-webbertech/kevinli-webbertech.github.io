# jenkins-deployment.yaml

Deployment manifest that runs Jenkins LTS with an `emptyDir` home volume and exposes the web UI (8080) and agent (50000) ports.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jenkins
spec:
  replicas: 1
  selector:
    matchLabels: { app: jenkins }
  template:
    metadata:
      labels: { app: jenkins }
    spec:
      containers:
        - name: jenkins
          image: jenkins/jenkins:lts-jdk17
          ports:
            - { containerPort: 8080 }
            - { containerPort: 50000 }
          volumeMounts:
            - { name: jenkins-home, mountPath: /var/jenkins_home }
      volumes:
        - name: jenkins-home
          emptyDir: {}
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/final_project_discussion.html)
