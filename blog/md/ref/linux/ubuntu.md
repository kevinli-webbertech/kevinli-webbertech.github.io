# Ubuntu System Commands

## systemd-analyze

The following commands will address the computer slowness issue.

`systemd-analyze critical-chain`

`systemd-analyze blame`

## Optimize the restart speed

* Stop all these services

`systemctl disable docker.service`

`systemctl disable mysql.service`

`systemctl disable NetworkManager-wait-online.service`

* Remove Snap

`sudo apt purge snap`

`sudo apt autoremove`

* Inspect the following [TODO]

```
55.619s plymouth-quit-wait.service
46.580s e2scrub_reap.service
33.979s podman-restart.service
```

* Frozen Terminal

Get `Activity Monitor` and kill `Gnome-shell`, it is like windows explorer and it will restart.


* Other

"$XDG_CURRENT_DESKTOP" "$GDMSESSION"

```
Ubuntu 18.04 and 20.04 (Ubuntu on GNOME)
XDG_CURRENT_DESKTOP=ubuntu:GNOME
GDMSESSION=ubuntu
Ubuntu 18.04 (Ubuntu on Wayland)
XDG_CURRENT_DESKTOP=ubuntu:GNOME
GDMSESSION=ubuntu-wayland
```

## Change Wayland to X11 Window Manager

* If you wish to do it permanently, edit /etc/gdm3/custom.conf and uncomment the line:
#WaylandEnable=false
* Save and reboot