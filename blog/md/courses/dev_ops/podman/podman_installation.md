# Podman Installation

## **For Linux Distributions:**

* Ubuntu (20.10+) or Kaili.
```
    sudo apt update
    sudo apt install -y podman
```

* Arch Linux & Manjaro Linux. 

`sudo pacman -S podman`

* Alpine Linux. 

`sudo apk add podman`

* CentOS Stream (9+). 

`sudo dnf -y install podman`


## For macOS 

* Install Homebrew if not already installed and Install Podman. 
    
`brew install podman`

• Initialize and start the Podman machine (Podman on macOS relies on a Linux virtual machine): 

```
podman machine init
podman machine start
```

## For Windows: 

Podman on Windows utilizes the Windows Subsystem for Linux (WSL2). 

* Enable WSL2: Ensure WSL2 is enabled on your Windows system. 
* Install Podman Desktop: Download and run the installer from the official Podman Desktop website. This will set up Podman within a WSL2 environment. 
* Alternatively, you can install Podman using package managers like Chocolately, Scoop, or Winget if preferred. 

## Verification (after installation): 

You can verify the installation and check the Podman version using: [1]  

```
podman --version
podman info
podman pull nginx
```

## Troubleshooting



### Ref

[1] https://www.youtube.com/watch?v=VdbjFgqPPE8

