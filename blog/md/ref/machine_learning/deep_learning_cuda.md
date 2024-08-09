# CUDA

## Installation of drivers, toolkits and cuDNN

**CUDA toolkit**

`RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-rhel9-12-3-local-12.3.2_545.23.08-1.x86_64.rpm && \
    rpm -i cuda-repo-rhel9-12-3-local-12.3.2_545.23.08-1.x86_64.rpm && dnf clean all && dnf -y install cuda-toolkit-12-3`

**cuDNN lib**

`RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && \
    rpm -i cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && dnf clean all && dnf -y install cudnn-cuda-12`

### Ref

* https://www.tensorflow.org/install/pip#software_requirements

* https://developer.nvidia.com/cuda-toolkit-archive

* https://developer.nvidia.com/cudnn



ref:

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_local