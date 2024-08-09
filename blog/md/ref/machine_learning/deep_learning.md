# Deep Learning

## Software

* Tensorflow, CUDA
* Keras is part of Tensorflow now

## Hardware

### NAVIDIA GPU

`v100`: NVIDIA® V100 Tensor Core is the most advanced data center GPU ever built to accelerate AI, high performance computing (HPC), data science and graphics. It’s powered by NVIDIA Volta architecture, comes in 16 and 32GB configurations, and offers the performance of up to 32 CPUs in a single GPU.

V100 is engineered for the convergence of AI and HPC. It offers a platform for HPC systems to excel at both computational science for scientific simulation and data science for finding insights in data. By pairing NVIDIA CUDA® cores and Tensor Cores within a unified architecture, a single server with V100 GPUs can replace hundreds of commodity CPU-only servers for both traditional HPC and AI workloads. Every researcher and engineer can now afford an AI supercomputer to tackle their most challenging work.

![v100_tensor_core_gpu](../../../images/ml/v100_tensor_core_gpu.png)

* https://www.nvidia.com/en-us/data-center/v100/

**Data Center GPU**

A2, A10, A16, A30, A40, T4, L4, L40s, L40, H100, H200, GB200 NVL2, GB200 NVL72.

**NVIDIA Multi-Instance GPU**

Seven independent instances in a single GPU.

![MIG](../../../images/ml/MIG.png)

`Multi-Instance GPU (MIG)` expands the performance and value of NVIDIA Blackwell and Hopper™ generation GPUs. MIG can partition the GPU into as many as seven instances, each fully isolated with its own high-bandwidth memory, cache, and compute cores. This gives administrators the ability to support every workload, from the smallest to the largest, with guaranteed quality of service (QoS) and extending the reach of accelerated computing resources to every user.

Without MIG, different jobs running on the same GPU, such as different AI inference requests, compete for the same resources. A job consuming larger memory bandwidth starves others, resulting in several jobs missing their latency targets. With MIG, jobs run simultaneously on different instances, each with dedicated resources for compute, memory, and memory bandwidth, resulting in predictable performance with QoS and maximum GPU utilization.

* Provision and Configure Instances as Needed
* Run Workloads in Parallel, Securely

**NVIDIA Blackwell**

**Hopper™**

### Ref

* https://aws.amazon.com/ec2/instance-types/p3/

P3 Docker image (NVidia V100) that supports all the P3 instances.

P4, P5, G3, etc., then a new image would be needed.