# nexus-deployment.yaml

Deployment manifest that runs Sonatype Nexus 3 with an `emptyDir` data volume and exposes its port 8081.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus
spec:
  replicas: 1
  selector:
    matchLabels: { app: nexus }
  template:
    metadata:
      labels: { app: nexus }
    spec:
      containers:
        - name: nexus
          image: sonatype/nexus3:latest
          ports: [{ containerPort: 8081 }]
          volumeMounts:
            - { name: nexus-data, mountPath: /nexus-data }
      volumes:
        - name: nexus-data
          emptyDir: {}
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/final_project_discussion.html)
