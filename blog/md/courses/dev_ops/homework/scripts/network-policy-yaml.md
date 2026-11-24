# network-policy.yaml

NetworkPolicy that explicitly permits Jenkins-to-Nexus traffic on TCP 8081 and Jenkins/webapp-to-MySQL traffic on TCP 3306, plus inbound Jenkins UI/agent ports, for CNI plug-ins that enforce NetworkPolicy.

```yaml
# Kubernetes Pods already share the cluster network. This policy is only enforced
# by CNI plug-ins that support NetworkPolicy; it explicitly permits required traffic.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ci-and-database-traffic
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: jenkins
      ports:
        - protocol: TCP
          port: 8081
    - from:
        - podSelector:
            matchLabels:
              app: webapp
        - podSelector:
            matchLabels:
              app: jenkins
      ports:
        - protocol: TCP
          port: 3306
    - ports:
        - protocol: TCP
          port: 8080
        - protocol: TCP
          port: 50000
```

[Back to Final Project Discussion](https://kevinli-webbertech.github.io/blog/html/courses/dev_ops/k8s/final_project_discussion.html)
