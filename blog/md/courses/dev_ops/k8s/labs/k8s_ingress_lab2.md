## K8s Ingress Lab2: Deploy Sample Apps

**Why?**

We'll simulate a **multi-service app** (e.g., a website with a blog and a store).

First we need to deploy the two apps we'll be using.

`kubectl create deployment webapp1 --image=gcr.io/google-sample/hello-app:1.0`

`kubectl create deployment webapp2 --image=gcr.io/google-sample/hello-app:2.0`

![Deploying the apps](/blog/images/dev_ops/k8s_ingress/deploy_apps.PNG)


* Make sure the ingress service is on

`minikube addons list`

We then have to expose them internally.

`kubectl expose deployment webapp1 --port=8080 --target-port=8080`

`kubectl expose deployment webapp2 --port=8080 --target-port=8080`

![Exposing the apps](/blog/images/dev_ops/k8s_ingress/expose_apps.PNG)

**Key Points**
- expose creates ClusterIP Services (internal DNS names webapp 1 and webapp 2).
- Without Ingress, these are only accessible inside the cluster.

## Configure Ingress

**Why?**

We'll define rules like:
- myapp.local/blog -> webapp1
- myapp.local/shop -> webapp2

This is how companies route paths/subdomains to different teams' services.

![Entering the configuration](/blog/images/dev_ops/k8s_ingress/nano_ingress.PNG)

`touch ingress.yaml`

Paste:

```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: example-ingress
spec:
  rules:
  - host: "myapp.local"  # Fake domain for testing
    http:
      paths:
      - path: /blog      # Route /blog to webapp1
        pathType: Prefix
        backend:
          service:
            name: webapp1
            port:
              number: 8080
      - path: /shop      # Route /shop to webapp2
        pathType: Prefix
        backend:
          service:
            name: webapp2
            port:
              number: 8080
```



![Inside of ingress](/blog/images/dev_ops/k8s_ingress/ingress_inside.PNG)


Make sure to then apply the settings.


`kubectl apply -f ingress.yaml`

![Applying the changes of ingress](/blog/images/dev_ops/k8s_ingress/nano_ingress_apply.PNG)

Lastly verify if everything went through.

![Verification](/blog/images/dev_ops/k8s_ingress/verify_ingress.PNG)

## Step 4: Test Routing

**Why?**

We need to confirm traffic is routed correctly (just like debugging a real website).

Get the Ingress IP (Minikube)

![Ingress IP](/blog/images/dev_ops/k8s_ingress/ingress_ip.PNG)

Add to /etc/hosts (simulate DNS)

`echo "$(minikube ip) myapp.local" |sudo tee -a /etc/hosts`

![Simulate DNS](/blog/images/dev_ops/k8s_ingress/simulate_DNS.PNG)

**Test**

![Testing](/blog/images/dev_ops/k8s_ingress/ingress_test.PNG)

**Key Points**
- /etc/hosts trick your computer into thinking myapp.local points to Minikube
- Ingress reads the Host header and path to route requests (like a web server's virtual hosts).

## Part 5: Add HTTPS (TLS)

**Why?** 

Production apps **must** use HTTPS. Ingress centralizes TLS termination (no certs in Pods).

Generate a self-signed cert (real apps use Let's Encrypt)

![Generating a self-signed cert](/blog/images/dev_ops/k8s_ingress/cert_generate.PNG)

Store cert in kubernetes 

![Storing the cert in Kubernetes](/blog/images/dev_ops/k8s_ingress/store_cert_K8s.PNG)

Update the ingress.yaml to use TLS

![Update of ingress.yaml](/blog/images/dev_ops/k8s_ingress/nano_ingress_update.PNG)

Add:
spec:
  tls:
  - hosts:
    - myapp.local
    secretName: myapp-tls

![Updated version of ingress.yaml](/blog/images/dev_ops/k8s_ingress/ingress_inside_update.PNG)

Don't forget to apply the changes.

![Application of changes](/blog/images/dev_ops/k8s_ingress/nano_ingress_apply_update.PNG)

### Test HTTPS

![Testing https](/blog/images/dev_ops/k8s_ingress/test_https.PNG)

 -k ignores self-signed cert warnings

## Part 6: Clean Up

**Why?**

Avoid leaving resources running (especially in cloud environments where they cost money).

![Cleanup](/blog/images/dev_ops/k8s_ingress/cleanup.PNG)


## Real-World Summary

Ingress = Traffic router (like NGINX for Kubernetes).
Why Use it?
- Path-based routing (/api, /app).
- Host-based routing (blog.example.com, shop.example.com).
- Centralized TLS (no cert management in Pods).
  - Used by companies like Spotify, Airbnb to manage microservices.
