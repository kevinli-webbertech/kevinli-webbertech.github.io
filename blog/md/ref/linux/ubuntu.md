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