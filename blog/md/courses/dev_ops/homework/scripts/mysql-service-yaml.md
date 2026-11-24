# mysql-service.yaml

ClusterIP service that gives Jenkins and the web application a stable `mysql-service:3306` DNS name for the database.

```yaml
apiVersion: v1
kind: Service
metadata: { name: mysql-service }
spec:
  selector: { app: mysql }
  ports: [{ port: 3306, targetPort: 3306 }]
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/final_project_discussion.html)
