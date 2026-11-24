# mysql-pvc.yaml

PersistentVolumeClaim requesting 5Gi of `ReadWriteOnce` storage for MySQL.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: mysql-pvc }
spec:
  accessModes: [ReadWriteOnce]
  resources: { requests: { storage: 5Gi } }
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/k8s/final_project_discussion.html)
