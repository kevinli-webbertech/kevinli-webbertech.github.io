# CUDA

## CentOS notes

Red Hat Enterprise Linux (RHEL) 9 is the first major release built using CentOS Stream, an open source development platform for RHEL. CentOS Stream is where community members can collaborate with Red Hat developers to create, test, and contribute to future versions of RHEL. CentOS Stream 9 is the upstream for the next minor release of RHEL 9. 
CentOS Linux is a community-supported distribution of Linux that's derived from Red Hat's source code. CentOS Linux is similar to RHEL in terms of functionality, compatibility, and bug fixes, and historically each version of CentOS Linux reflected a major version of RHEL. However, Red Hat discontinued updates to CentOS Linux between 2021 and 2024, and its end-of-support date is June 30, 2024.

## Intro of CUDA

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

## Installation of drivers, toolkits and cuDNN

* https://www.tensorflow.org/install/pip#software_requirements

**Nvidia driver**

https://www.nvidia.com/en-us/drivers/details/226775/

**gcc**

gcc --version

**cuda toolkit(driver)**

`RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-rhel9-12-3-local-12.3.2_545.23.08-1.x86_64.rpm && \
    rpm -i cuda-repo-rhel9-12-3-local-12.3.2_545.23.08-1.x86_64.rpm && dnf clean all && dnf -y install cuda-toolkit-12-3`

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9 (cuda 12.6)

https://developer.nvidia.com/cuda-toolkit-archive (12.3)

**Troubleshooting**

```shell
yum -y install pciutils
lspci | grep -e VGA -ie NVIDIA
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
nano /home/UserName/.bash_profile
CUDA_HOME=/usr/local/cuda-10.1
LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export CUDA_HOME
export LD_LIBRARY_PATH
```

Using built-in stream user interface
-> Detected 4 CPUs online; setting concurrency level to 4.
-> Unable to locate any tools for listing initramfs contents.
-> Unable to scan initramfs: no tool found
ERROR: Unable to find the module utility `modprobe`; please make sure you have the package 'module-init-tools' or 'kmod' installed.  If you do have 'module-init-tools' or 'kmod' installed, then please check that `modprobe` is in your PATH.

* https://www.dell.com/support/kbdoc/en-us/000216077/how-to-install-nvidia-driver-in-rhel

* https://medium.com/analytics-vidhya/installing-cuda-on-red-hat-linux-and-ubuntu-dda69cd5ab9c

**cuDNN lib**

`RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && \
    rpm -i cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && dnf clean all && dnf -y install cudnn-cuda-12`

**OpenGL** (optional)

http://repo.okay.com.mx/centos/8/x86_64/release/libglvnd-opengl-1.0.1-0.9.git5baa1e5.el8.x86_64.rpm

Install OKey repository:

libglvnd-1.3.4-1.el9.x86_64.rpm

wget https://dl.rockylinux.org/pub/rocky/9/devel/x86_64/os/Packages/l/libglvnd-1.3.4-1.el9.x86_64.rpm (working)

wget https://rpmfind.net/linux/centos-stream/9-stream/AppStream/x86_64/os/Packages/libglvnd-opengl-1.3.4-1.el9.x86_64.rpm (working)

Install libglvnd-opengl  rpm package:

`# dnf install libglvnd-opengl (not executed)`

**Test the Nvidia Installation**

* After the Nvidia drivers are installed, you can test the installation by running the command:

`nvcc -V`
Type `nvcc --version` and press Enter.

or

`cat /usr/local/cuda/version.txt`

`nvidia-smi`

### Ref

* https://developer.nvidia.com/cuda-toolkit-archive

* https://developer.nvidia.com/cudnn

* https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_local