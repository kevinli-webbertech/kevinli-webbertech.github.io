# mysql-configmap.yaml

ConfigMap holding the non-secret MySQL database name and username.

```yaml
apiVersion: v1
kind: ConfigMap
metadata: { name: mysql-config }
data:
  MYSQL_DATABASE: myappdb
  MYSQL_USER: myappuser
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/final_project_discussion.html)
