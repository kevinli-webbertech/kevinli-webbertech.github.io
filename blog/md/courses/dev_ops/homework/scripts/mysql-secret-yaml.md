# mysql-secret.yaml

Opaque secret for the MySQL root and application passwords. Replace both placeholder passwords before using this outside a local demonstration.

```yaml
apiVersion: v1
kind: Secret
metadata: { name: mysql-secret }
type: Opaque
stringData:
  MYSQL_ROOT_PASSWORD: ChangeRootPassword123!
  MYSQL_PASSWORD: ChangeUserPassword123!
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/k8s/final_project_discussion.html)
