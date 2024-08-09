# CUDA

## Installation of drivers, toolkits and cuDNN

**CUDA toolkit**

`RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-rhel9-12-3-local-12.3.2_545.23.08-1.x86_64.rpm && \
    rpm -i cuda-repo-rhel9-12-3-local-12.3.2_545.23.08-1.x86_64.rpm && dnf clean all && dnf -y install cuda-toolkit-12-3`

bash-5.1# dnf -y install cuda-toolkit-12-3
cuda-rhel9-12-3-local                                                                                                             642 kB/s | 102 kB     00:00    
Red Hat Universal Base Image 9 (RPMs) - BaseOS                                                                                    1.1 MB/s | 516 kB     00:00    
Red Hat Universal Base Image 9 (RPMs) - AppStream                                                                                 4.4 MB/s | 2.1 MB     00:00    
Red Hat Universal Base Image 9 (RPMs) - CodeReady Builder                                                                         712 kB/s | 275 kB     00:00    
Error: 
 Problem: package cuda-tools-12-3-12.3.2-1.x86_64 from cuda-rhel9-12-3-local requires cuda-visual-tools-12-3 >= 12.3.2, but none of the providers can be installed
  - package cuda-visual-tools-12-3-12.3.2-1.x86_64 from cuda-rhel9-12-3-local requires cuda-nsight-systems-12-3 >= 12.3.2, but none of the providers can be installed
  - package cuda-toolkit-12-3-12.3.2-1.x86_64 from cuda-rhel9-12-3-local requires cuda-tools-12-3 >= 12.3.2, but none of the providers can be installed
  - package cuda-nsight-systems-12-3-12.3.2-1.x86_64 from cuda-rhel9-12-3-local requires nsight-systems >= 2023.3.3.42, but none of the providers can be installed
  - conflicting requests
  - nothing provides libxcb-icccm.so.4()(64bit) needed by nsight-systems-2023.3.3-2023.3.3.42_233333266658v0-0.x86_64 from cuda-rhel9-12-3-local
  - nothing provides libxcb-image.so.0()(64bit) needed by nsight-systems-2023.3.3-2023.3.3.42_233333266658v0-0.x86_64 from cuda-rhel9-12-3-local
  - nothing provides libxcb-keysyms.so.1()(64bit) needed by nsight-systems-2023.3.3-2023.3.3.42_233333266658v0-0.x86_64 from cuda-rhel9-12-3-local
  - nothing provides libxcb-render-util.so.0()(64bit) needed by nsight-systems-2023.3.3-2023.3.3.42_233333266658v0-0.x86_64 from cuda-rhel9-12-3-local
  - nothing provides libxkbcommon-x11.so.0()(64bit) needed by nsight-systems-2023.3.3-2023.3.3.42_233333266658v0-0.x86_64 from cuda-rhel9-12-3-local
  - nothing provides libOpenGL.so.0()(64bit) needed by nsight-systems-2023.3.3-2023.3.3.42_233333266658v0-0.x86_64 from cuda-rhel9-12-3-local
(try to add '--skip-broken' to skip uninstallable packages or '--nobest' to use not only best candidate packages)

**cuDNN lib**

`RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && \
    rpm -i cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && dnf clean all && dnf -y install cudnn-cuda-12`

### Ref

* https://www.tensorflow.org/install/pip#software_requirements

* https://developer.nvidia.com/cuda-toolkit-archive

* https://developer.nvidia.com/cudnn

## Ref

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_local