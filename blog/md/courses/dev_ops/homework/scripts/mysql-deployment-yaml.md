# mysql-deployment.yaml

MySQL 8.4 deployment that reads its configuration from `mysql-config` and `mysql-secret`, and persists data on the `mysql-pvc` claim.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: mysql }
spec:
  strategy: { type: Recreate }
  selector: { matchLabels: { app: mysql } }
  template:
    metadata: { labels: { app: mysql } }
    spec:
      containers:
        - name: mysql
          image: mysql:8.4
          ports: [{ containerPort: 3306 }]
          envFrom: [{ configMapRef: { name: mysql-config } }, { secretRef: { name: mysql-secret } }]
          volumeMounts: [{ name: mysql-storage, mountPath: /var/lib/mysql }]
      volumes: [{ name: mysql-storage, persistentVolumeClaim: { claimName: mysql-pvc } }]
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/k8s/final_project_discussion.html)
