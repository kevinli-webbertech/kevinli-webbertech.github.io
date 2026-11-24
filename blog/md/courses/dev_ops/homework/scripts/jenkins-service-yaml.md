# jenkins-service.yaml

NodePort service that exposes the Jenkins web UI on port 32000 and the agent port on 50000.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: jenkins
spec:
  type: NodePort
  selector: { app: jenkins }
  ports:
    - { name: http, port: 8080, targetPort: 8080, nodePort: 32000 }
    - { name: agent, port: 50000, targetPort: 50000 }
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/k8s/final_project_discussion.html)
