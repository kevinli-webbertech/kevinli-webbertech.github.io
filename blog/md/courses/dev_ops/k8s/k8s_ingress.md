# Introduction to Kubernetes Ingress

## What is Ingress?

In Kubernetes, **Ingress** is an API object that manages external access to services in a cluster,
typically HTTP/HTTPS traffic. It acts as a **smart traffic router**, directing requests to the right
services based on hostnames, path, or other rules.

## Why Use Ingress? (Real-World Analogy)

Think of Ingress like a **restaurant host**:

- **Without Ingress**: You have to know exactly which table (Pod) to go to --this is like using
NodePort or LoadBalancer services directly.
- **With Ingress**: The host (Ingress) takes your request ("I want the Web App Menu") and directs
you to the right table (Service) based on predefined rules.

## Common Use Cases

1. **Host-Based Routing**: Serve multiple websites (e.g., blog.example.com, shop.example.com) 
from the same cluster.
2. **Path-Based Routing**: Different paths (/api, /app) route to different backend services.
3. **TLS/SSL Termination**: Handle HTTPS encryption at the Ingress level instead of inside Pods.
4. **Load Balancing**: Distribute traffic across multiple backend services.

# Lab Setup

## Install Prerequisites

Ensure you have kubectl, a Kubernetes cluster (Minikube for local testing, and curl for testing.

**Install kubectl (if not installed)**

![system update](/blog/images/dev_ops/k8s_ingress/System_Update.PNG)

![GPG Key](/blog/images/dev_ops/k8s_ingress/K8s_GPGKey.PNG)

![K8s APT Repository](/blog/images/dev_ops/k8s_ingress/K8s_APT_Repository.PNG)

![Kubectl install](/blog/images/dev_ops/k8s_ingress/kubectl_install.PNG)

**Install Minikube (for local testing)**

![Minikube install](/blog/images/dev_ops/k8s_ingress/minikube_install.PNG)

![Minikube install 2](/blog/images/dev_ops/k8s_ingress/minikube_install2.PNG)

## Part 1: Enable Ingress Controller

For minikube (simplest option)

![Starting Minikube](/blog/images/dev_ops/k8s_ingress/minikube_start.PNG)

**Why?**

- Real clusters (EKS,GKE) cost money. Minikube emulates a cluster on your laptop.
- --driver=docker uses Docker containers as "virtual nodes" (lightweight)

![Enabling the Ingress Controller](/blog/images/dev_ops/k8s_ingress/enable_ingress_controller.PNG)

**Why?**

- Ingress isn't enabled by default (like how AWS/GKE require you to install Nginx).
- This deploys an **Nginx Ingress Controller** Pod in your cluster, which acts as the "traffic cop" for routing.

**Verify**

![Verifying the process of enabling the ingress controller](/blog/images/dev_ops/k8s_ingress/verify_controller.PNG)

You should see ingress-nginx-controller-xxxx running.

## Part 2: Deploy Sample Apps

**Why?**

We'll simulate a **multi-service app** (e.g., a website with a blog and a store).

First we need to deploy the two apps we'll be using.

![Deploying the apps](/blog/images/dev_ops/k8s_ingress/deploy_apps.PNG)


We then have to expose them internally.

![Exposing the apps](/blog/images/dev_ops/k8s_ingress/expose_apps.PNG)

**Key Points**
- expose creates ClusterIP Services (internal DNS names webapp 1 and webapp 2).
- Without Ingress, these are only accessible inside the cluster.

## Part 3: Configure Ingress

**Why?**

We'll define rules like:
- myapp.local/blog -> webapp1
- myapp.local/shop -> webapp2

This is how companies route paths/subdomains to different teams' services.

![Entering the configuration](/blog/images/dev_ops/k8s_ingress/nano_ingress.PNG)

Paste:
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

![Inside of ingress](/blog/images/dev_ops/k8s_ingress/ingress_inside.PNG)


Make sure to then apply the settings.

![Applying the changes of ingress](/blog/images/dev_ops/k8s_ingress/nano_ingress_apply.PNG)

Lastly verify if everything went through.

![Verification](/blog/images/dev_ops/k8s_ingress/verify_ingress.PNG)

## Step 4: Test Routing

**Why?**

We need to confirm traffic is routed correctly (just like debugging a real website).

Get the Ingress IP (Minikube)

![Ingress IP](/blog/images/dev_ops/k8s_ingress/ingress_ip.PNG)

Add to /etc/hosts (simulate DNS)

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
