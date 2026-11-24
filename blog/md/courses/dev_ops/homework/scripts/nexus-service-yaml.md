# nexus-service.yaml

NodePort service that exposes the Nexus repository manager on port 32081.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nexus
spec:
  type: NodePort
  selector: { app: nexus }
  ports:
    - { port: 8081, targetPort: 8081, nodePort: 32081 }
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/homework/final_project_discussion.html)
