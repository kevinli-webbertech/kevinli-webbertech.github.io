# Install Oracle Virtualbox in Ubuntu 22

Got the deb package from oracle, and `chmod +x *.deb`

```
xiaofengli@xiaofenglx:~/Downloads$ sudo dpkg -i virtualbox-7.0_7.0.18-162988~Ubuntu~jammy_amd64.deb 
[sudo] password for xiaofengli: 
Selecting previously unselected package virtualbox-7.0.
(Reading database ... 249268 files and directories currently installed.)
Preparing to unpack virtualbox-7.0_7.0.18-162988~Ubuntu~jammy_amd64.deb ...
Unpacking virtualbox-7.0 (7.0.18-162988~Ubuntu~jammy) ...
dpkg: dependency problems prevent configuration of virtualbox-7.0:
 virtualbox-7.0 depends on libqt5help5 (>= 5.15.1); however:
  Package libqt5help5 is not installed.
 virtualbox-7.0 depends on libqt5opengl5 (>= 5.0.2); however:
  Package libqt5opengl5 is not installed.
 virtualbox-7.0 depends on libqt5printsupport5 (>= 5.0.2); however:
  Package libqt5printsupport5 is not installed.
 virtualbox-7.0 depends on libqt5xml5 (>= 5.0.2); however:
  Package libqt5xml5 is not installed.

dpkg: error processing package virtualbox-7.0 (--install):
 dependency problems - leaving unconfigured
Processing triggers for mailcap (3.70+nmu1ubuntu1) ...
Processing triggers for gnome-menus (3.36.0-1ubuntu3) ...
Processing triggers for desktop-file-utils (0.26-1ubuntu3) ...
Processing triggers for hicolor-icon-theme (0.17-2) ...
Processing triggers for shared-mime-info (2.1-2) ...
Errors were encountered while processing:
 virtualbox-7.0

 xiaofengli@xiaofenglx:~/Downloads$ sudo apt --fix-broken install
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Correcting dependencies... Done
The following packages were automatically installed and are no longer required:
  python3-brlapi python3-louis python3-pyatspi python3-speechd xbrlapi
Use 'sudo apt autoremove' to remove them.
The following packages will be REMOVED:
  virtualbox-7.0
0 upgraded, 0 newly installed, 1 to remove and 135 not upgraded.
1 not fully installed or removed.
After this operation, 220 MB disk space will be freed.
Do you want to continue? [Y/n] Y
(Reading database ... 250007 files and directories currently installed.)
Removing virtualbox-7.0 (7.0.18-162988~Ubuntu~jammy) ...
Processing triggers for hicolor-icon-theme (0.17-2) ...
Processing triggers for gnome-menus (3.36.0-1ubuntu3) ...
Processing triggers for shared-mime-info (2.1-2) ...
Processing triggers for mailcap (3.70+nmu1ubuntu1) ...
Processing triggers for desktop-file-utils (0.26-1ubuntu3) ...

xiaofengli@xiaofenglx:~/Downloads$ sudo apt install libqt5xml5
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following packages were automatically installed and are no longer required:
  python3-brlapi python3-louis python3-pyatspi python3-speechd xbrlapi
Use 'sudo apt autoremove' to remove them.
The following NEW packages will be installed:
  libqt5xml5
0 upgraded, 1 newly installed, 0 to remove and 135 not upgraded.
Need to get 144 kB of archives.
After this operation, 476 kB of additional disk space will be used.
Get:1 https://deepin-wine.i-m.dev  libqt5xml5 5.15.8.1-1+dde [144 kB]
Fetched 144 kB in 2s (66.5 kB/s)                                
Selecting previously unselected package libqt5xml5:amd64.
(Reading database ... 249268 files and directories currently installed.)
Preparing to unpack .../libqt5xml5_5.15.8.1-1+dde_amd64.deb ...
Unpacking libqt5xml5:amd64 (5.15.8.1-1+dde) ...
Setting up libqt5xml5:amd64 (5.15.8.1-1+dde) ...
Processing triggers for libc-bin (2.35-0ubuntu3.7) ...

```

Rerun

```
xiaofengli@xiaofenglx:~/Downloads$ sudo dpkg -i virtualbox-7.0_7.0.18-162988~Ubuntu~jammy_amd64.deb 
Selecting previously unselected package virtualbox-7.0.
(Reading database ... 249275 files and directories currently installed.)
Preparing to unpack virtualbox-7.0_7.0.18-162988~Ubuntu~jammy_amd64.deb ...
Unpacking virtualbox-7.0 (7.0.18-162988~Ubuntu~jammy) ...
dpkg: dependency problems prevent configuration of virtualbox-7.0:
 virtualbox-7.0 depends on libqt5help5 (>= 5.15.1); however:
  Package libqt5help5 is not installed.
 virtualbox-7.0 depends on libqt5opengl5 (>= 5.0.2); however:
  Package libqt5opengl5 is not installed.
 virtualbox-7.0 depends on libqt5printsupport5 (>= 5.0.2); however:
  Package libqt5printsupport5 is not installed.

dpkg: error processing package virtualbox-7.0 (--install):
 dependency problems - leaving unconfigured
Processing triggers for mailcap (3.70+nmu1ubuntu1) ...
Processing triggers for gnome-menus (3.36.0-1ubuntu3) ...
Processing triggers for desktop-file-utils (0.26-1ubuntu3) ...
Processing triggers for hicolor-icon-theme (0.17-2) ...
Processing triggers for shared-mime-info (2.1-2) ...

xiaofengli@xiaofenglx:~/Downloads$ sudo apt install libqt5help5
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
You might want to run 'apt --fix-broken install' to correct these.
The following packages have unmet dependencies:
 libqt5help5 : Depends: libqt5sql5 (>= 5.3.0) but it is not going to be installed
 virtualbox-7.0 : Depends: libqt5opengl5 (>= 5.0.2) but it is not going to be installed
                  Depends: libqt5printsupport5 (>= 5.0.2) but it is not going to be installed
                  Recommends: libsdl-ttf2.0-0 but it is not going to be installed
E: Unmet dependencies. Try 'apt --fix-broken install' with no packages (or specify a solution).

```

Need to make sure the following are installed, 

dpkg: dependency problems prevent configuration of virtualbox-7.0:
 virtualbox-7.0 depends on libqt5help5 (>= 5.15.1); however:
  Package libqt5help5 is not installed.
 virtualbox-7.0 depends on libqt5opengl5 (>= 5.0.2); however:
  Package libqt5opengl5 is not installed.
 virtualbox-7.0 depends on libqt5printsupport5 (>= 5.0.2); however:
  Package libqt5printsupport5 is not installed.


## TODO

