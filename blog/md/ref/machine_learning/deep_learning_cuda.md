# CUDA

## CentOS notes

Red Hat Enterprise Linux (RHEL) 9 is the first major release built using CentOS Stream, an open source development platform for RHEL. CentOS Stream is where community members can collaborate with Red Hat developers to create, test, and contribute to future versions of RHEL. CentOS Stream 9 is the upstream for the next minor release of RHEL 9. 
CentOS Linux is a community-supported distribution of Linux that's derived from Red Hat's source code. CentOS Linux is similar to RHEL in terms of functionality, compatibility, and bug fixes, and historically each version of CentOS Linux reflected a major version of RHEL. However, Red Hat discontinued updates to CentOS Linux between 2021 and 2024, and its end-of-support date is June 30, 2024.

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


cuda-tools-12-3
nsight-systems-2023.3.3-2023.3.3.42_233333266658v0-0.x86_64

libxcb-icccm.so.4 ([xcb-util-wm-0.4.1-12.el8.x86_64.rpm](https://rhel.pkgs.org/8/okey-x86_64/xcb-util-wm-0.4.1-12.el8.x86_64.rpm.html))
libxcb-image.so.0 ([xcb-util-image-0.4.0-9.el8.x86_64.rpm](https://rhel.pkgs.org/8/okey-x86_64/xcb-util-image-0.4.0-9.el8.x86_64.rpm.html))
libxcb-keysyms.so.1 ([xcb-util-keysyms-0.4.0-7.el8.x86_64.rpm](https://rhel.pkgs.org/8/okey-x86_64/xcb-util-keysyms-0.4.0-7.el8.x86_64.rpm.html))
libxcb-render-util.so.0 (https://rhel.pkgs.org/8/okey-x86_64/xcb-util-renderutil-0.3.9-10.el8.x86_64.rpm.html)
libxkbcommon-x11.so.0 (https://rhel.pkgs.org/8/raven-x86_64/libxkbcommon-x11-0.9.1-1.el8.x86_64.rpm.html)
libOpenGL.so.0(https://rhel.pkgs.org/8/okey-x86_64/libglvnd-opengl-1.0.1-0.9.git5baa1e5.el8.x86_64.rpm.html)

**OpenGL**
http://repo.okay.com.mx/centos/8/x86_64/release/libglvnd-opengl-1.0.1-0.9.git5baa1e5.el8.x86_64.rpm

Install OKey repository:

`# dnf install http://repo.okay.com.mx/centos/8/x86_64/release/libglvnd-opengl-1.0.1-0.9.git5baa1e5.el8.x86_64.rpm --skip-broken`


Install libglvnd-opengl  rpm package:

`# dnf install libglvnd-opengl (not executed)`


bash-5.1# rpm -i libglvnd-opengl-1.0.1-0.9.git5baa1e5.el8.x86_64.rpm
warning: libglvnd-opengl-1.0.1-0.9.git5baa1e5.el8.x86_64.rpm: Header V4 RSA/SHA256 Signature, key ID 186f7970: NOKEY
error: Failed dependencies:
	libGLdispatch.so.0()(64bit) is needed by libglvnd-opengl-1:1.0.1-0.9.git5baa1e5.el8.x86_64
	libglvnd(x86-64) = 1:1.0.1-0.9.git5baa1e5.el8 is needed by libglvnd-opengl-1:1.0.1-0.9.git5baa1e5.el8.x86_64


**cuDNN lib**

`RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && \
    rpm -i cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm && dnf clean all && dnf -y install cudnn-cuda-12`

### Ref

* https://www.tensorflow.org/install/pip#software_requirements

* https://developer.nvidia.com/cuda-toolkit-archive

* https://developer.nvidia.com/cudnn

## Ref

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_local