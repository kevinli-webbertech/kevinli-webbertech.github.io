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

NVIDIA Blackwell Architecture: the NVIDIA Blackwell architecture brings to generative AI and accelerated computing. It is an architecture to build AI superchip.

Blackwell-architecture GPUs pack 208 billion transistors and are manufactured using a custom-built TSMC 4NP process. All Blackwell products feature two reticle-limited dies connected by a 10 terabytes per second (TB/s) chip-to-chip interconnect in a unified single GPU.

The second-generation Transformer Engine uses custom Blackwell Tensor Core technology combined with NVIDIA® TensorRT™-LLM and NeMo™ Framework innovations to accelerate inference and training for large language models (LLMs) and Mixture-of-Experts (MoE) models.

Blackwell includes NVIDIA Confidential Computing, which protects sensitive data and AI models from unauthorized access with strong hardware-based security.

Unlocking the full potential of exascale computing and trillion-parameter AI models hinges on the need for swift, seamless communication among every GPU within a server cluster. The fifth-generation of `NVIDIA® NVLink®` interconnect can scale up to 576 GPUs to unleash accelerated performance for trillion- and multi-trillion parameter AI models.  

The `NVIDIA NVLink Switch Chip` enables 130TB/s of GPU bandwidth in one 72-GPU NVLink domain (NVL72) and delivers 4X bandwidth efficiency with NVIDIA Scalable Hierarchical Aggregation and Reduction Protocol (SHARP)™ FP8 support.

The `NVIDIA GB200 NVL72` connects 36 GB200 Grace Blackwell Superchips with 36 Grace CPUs and 72 Blackwell GPUs in a rack-scale design. The GB200 NVL72 is a liquid-cooled solution with a 72-GPU NVLink domain that acts as a single massive GPU—delivering 30X faster real-time inference for trillion-parameter large language models.

**NVIDIA NVLink** and **NVLink Switch**

`Fully Connect GPUs With NVIDIA NVLink and NVLink Switch`

NVLink is a 1.8TB/s bidirectional, direct GPU-to-GPU interconnect that scales multi-GPU input and output (IO) within a server. The NVIDIA NVLink Switch chips connect multiple NVLinks to provide all-to-all GPU communication at full NVLink speed within a single rack and between racks.

To enable high-speed, collective operations, each NVLink Switch has engines for NVIDIA Scalable Hierarchical Aggregation and Reduction Protocol (SHARP)™ for in-network reductions and multicast acceleration.

![NVLink](../../../images/ml/NVLink.png)

![NVLink Switch](../../../images/ml/NVLink_Switch.png)

* https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/

**Hopper™**

* https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/

### Ref

* https://aws.amazon.com/ec2/instance-types/p3/

P3 Docker image (NVidia V100) that supports all the P3 instances.

P4, P5, G3, etc., then a new image would be needed.