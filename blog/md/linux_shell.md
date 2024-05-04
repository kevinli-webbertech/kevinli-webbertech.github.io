# Linux Ref

## Linux file system commands

* `man`	Shows a command’s manual
* `echo`	Prints a message as a standard output
* `ls`	Lists a directory’s content
* `pwd`	Shows the current working directory’s path
* `cd`	Changes the working directory
* `mkdir`	Creates a new directory
* `rm`	Deletes a file
* `cp`	Copies files and directories, including their content
* `mv`	Moves or renames files and directories
* `touch`	Creates a new empty file
* `file`	Checks a file’s type
* `zip` and unzip	Creates and extracts a ZIP archive
* `tar`	Archives files without compression in a TAR format
* `nano`, vi, and jed	Edits a file with a text editor
* `history`	Lists previously run commands
* `date` Display system time
* `cal`	Displays a calendar in Terminal. (not builtin, need to install `ncal`)
* `calc` Calculator
* `tree` folder structure

# Profile, Configuration and Path

* `ln`	Links files or directories
* `alias` and `unalias`	Sets and removes an alias for a file or command
* `export` export definition of system variable
* `source` execute the system profile

## Permissions

* `sudo`	Runs a command as a superuser
* `su`	Runs programs in the current shell as another user
* `chmod`	Modifies a file’s read, write, and execute permissions
* `chown`	Changes a file, directory, or symbolic link’s ownership
* `useradd` and `userdel`	Creates and removes a user account

# Files Operations
* `cat`	Lists, combines, and writes a file’s content as a standard output
* `less` Read file
* `more`  Read file
* `head`	Displays a file’s first ten lines
* `tail`	Prints a file’s last ten lines
* `sort`	Reorders a file’s content
* `cut`	Sections and prints lines from a file
* `diff`	Compares two files’ content and their differences
* `tee`	Prints command outputs in Terminal and a file
* `locate`	Finds files in a system’s database
* `find`	Outputs a file or folder’s location
* `wc`    counting utility

### Examples and usages

#### more examples

`more -n 5 file.txt`: show the first 5 lines
`more +20 file.txt`: show starting from the 20th line
`more -f file.txt`: This option enables the continuous display of the contents of a file.

#### tee examples

`wc -l file1.txt|tee -a file2.txt`

file1.txt

Input: geek
       for
       geeks

file2.txt

Input:geeks
      for
      geeks
OUTPUT : 3 file1.txt

cat file2.txt
OUTPUT:
      geeks
      for
      geeks
      3 file1.txt

`wc -l file1.txt| tee file2.txt`
will overwrite the output to fil2.txt

# Text find, replace and regex

* `sed`	Finds, replaces, or deletes patterns in a file
* `awk`	Finds and manipulates patterns in a file
* `grep`	Searches a string within a file

## Package and software management (debian)

* `apt`
* `apt-get`	Manages Debian-based distros package libraries
* `dpkg`    dpkg is a medium-level tool to install, build, remove and manage Debian packages.

### apt and apt-get examples

The apt command line tool provides a higher-level user interface for end users with intuitive commands, resulting behaviors, and security features. In contrast, the command apt-get is a low-level interface that communicates more closely with core Linux processes. The apt command is a more user-friendly package manager than apt-get.

`sudo apt-get update`: 

Used in Linux-based operating systems to update the package lists for available software packages from the configured repositories.

`sudo apt-get upgrade`

This command is used to install the latest versions of the packages currently installed on the user’s system from the sources enumerated in /etc/apt/sources.list. The installed packages which have new packages available are retrieved and installed. You need to perform an update before the upgrade so that apt-get knows that new versions of packages are available.

`sudo apt-get install [package_name]`

This command is used to install or upgrade packages. 

`sudo apt-get install [package_name]`

Skipped

`sudo apt-get remove [package_name]`

Remove software but does not remove configuration files

`sudo apt-get purge [package_name]`

Remove software and remove configuration files

`sudo apt-get check`

This command is used to update the package cache and check for broken dependencies.

`sudo apt-get clean`

This command is used to keep our system clean and tidy. It removes all the cached package files that were downloaded due to the downloading of recent packages using `apt-get`.

`sudo apt-get autoremove`

Sometimes the packages which are automatically installed to satisfy the dependencies of other packages, are no longer needed then the autoremove command is used to remove this kind of packages.

`sudo apt-get list`

It also gives details (version, architecture and repository source) about package but only if package is available or installed in our system. `sudo apt-get list firefox`

To get more options, check with -h,

```
Most used commands:
  update - Retrieve new lists of packages
  upgrade - Perform an upgrade
  install - Install new packages (pkg is libc6 not libc6.deb)
  reinstall - Reinstall packages (pkg is libc6 not libc6.deb)
  remove - Remove packages
  purge - Remove packages and config files
  autoremove - Remove automatically all unused packages
  dist-upgrade - Distribution upgrade, see apt-get(8)
  dselect-upgrade - Follow dselect selections
  build-dep - Configure build-dependencies for source packages
  satisfy - Satisfy dependency strings
  clean - Erase downloaded archive files
  autoclean - Erase old downloaded archive files
  check - Verify that there are no broken dependencies
  source - Download source archives
  download - Download the binary package into the current directory
  changelog - Download and display the changelog for the given package
```

### dpkg examples
```
  install
           The package is selected for installation.

       hold
           A package marked to be on hold is kept on the same version, that is, no automatic new
           installs, upgrades or removals will be performed on them, unless these actions are
           requested explicitly, or are permitted to be done automatically with the --force-hold
           option.

       deinstall
           The package is selected for deinstallation (i.e. we want to remove all files, except
           configuration files).

       purge
           The package is selected to be purged (i.e. we want to remove everything from system
           directories, even configuration files).

       unknown
           The package selection is unknown.  A package that is also in a not-installed state, and
           with an ok flag will be forgotten in the next database store.
```

## System Information

* `df`	Displays the system’s overall disk space usage
* `du`	Checks a file or directory’s storage consumption
* `top`	Displays running processes and the system’s resource usage
* `vmstat`  Report virtual memory statistic
* `htop`	Works like top but with an interactive user interface (not default command)
* `ps`	Creates a snapshot of all running processes
* `uname`	Prints information about your machine’s kernel, name, and hardware
* `hostname`	Shows your system’s hostname
* `time`	Calculates commands’ execution time
* `systemctl`	Manages system services
* `watch`	Runs another command continuously
* `jobs`	Displays a shell’s running processes with their statuses
* `kill`	Terminates a running process
* `bg`
* `fg`
* `lsof`
* `which`
* `whomai`
* `uptime`
* `shutdown`	Turns off or restarts the system
* `reboot`  Reboot

### Examples

```
xiaofengli@xiaofenglx:~/code/codebank$ uname -a
Linux xiaofenglx 6.5.0-28-generic #29~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Apr  4 14:39:20 UTC 2 x86_64 x86_64 x86_64 GNU/Linux

xiaofengli@xiaofenglx:~$ who
xiaofengli tty2         2024-05-03 09:13 (tty2)

xiaofengli@xiaofenglx:~$ whoami
xiaofengli

xiaofengli@xiaofenglx:~$ vmstat
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 0  2      0 28084128 166644 2465256    0    0  1911    99  537 1266  7  3 12 78  0

xiaofengli@xiaofenglx:~$ free
               total        used        free      shared  buff/cache   available
Mem:        32700488     1971364    28066992      248572     2662132    30066188
Swap:        2097148           0     2097148

xiaofengli@xiaofenglx:~$ free -m
               total        used        free      shared  buff/cache   available
Mem:           31934        1966       27362         242        2604       29319
Swap:           2047           0        2047

xiaofengli@xiaofenglx:~/code/codebank$ du -m
1	./BloggerDev
1	./.git/branches
1	./.git/refs/heads
1	./.git/refs/tags
1	./.git/refs/remotes/origin
1	./.git/refs/remotes
1	./.git/refs
1	./.git/info
2	./.git/objects/97

xiaofengli@xiaofenglx:~/code/codebank$ df
Filesystem       1K-blocks       Used  Available Use% Mounted on
tmpfs              3270052       3624    3266428   1% /run
/dev/sda2        479079112  129910872  324758848  29% /
tmpfs             16350244      46012   16304232   1% /dev/shm
tmpfs                 5120          4       5116   1% /run/lock
efivarfs               128         35         89  28% /sys/firmware/efi/efivars
/dev/sdb1      11718752252 1749660476 9969091776  15% /mnt/ntfs
/dev/sda1           523244       6220     517024   2% /boot/efi
tmpfs              3270048        140    3269908   1% /run/user/1000

xiaofengli@xiaofenglx:~/code/codebank$ lsof| grep -i 8080
apache2   8072 8080 apache2           www-data  cwd   unknown                                         /proc/8072/task/8080/cwd (readlink: Permission denied)
apache2   8072 8080 apache2           www-data  rtd   unknown                                         /proc/8072/task/8080/root (readlink: Permission denied)
apache2   8072 8080 apache2           www-data  txt   unknown                                         /proc/8072/task/8080/exe (readlink: Permission denied)
apache2   8072 8080 apache2           www-data NOFD                                                   /proc/8072/task/8080/fd (opendir: Permission denied)

```

## Network

* `ping`	Checks the system’s network connectivity
* `wget`	Downloads files from a URL
* `curl`	Transmits data between servers using URLs
* `scp`	Securely copies files or directories to another system
* `rsync`	Synchronizes content between directories or machines
* `lfconfig`	Displays the system’s network interfaces and their configurations
* `netstat`	Shows the system’s network information, like routing and sockets
* `traceroute`	Tracks a packet’s hops to its destination
* `nslookup`	Queries a domain’s IP address and vice versa
* `dig`	Displays DNS information, including record types

### Examples

```
wget
```

```
curl
```

```
netstat
```

```
```

## Firewall & IPtable


## File and file descriptor


## Shell scripting

### function

### loop

### case

### redirect




## Ref

* https://man7.org/tlpi/index.html
* https://www.dpkg.org/
* https://manpages.ubuntu.com/manpages/trusty/man1/dpkg.1.html
* https://tldp.org/
* https://github.com/tLDP
* https://www.linuxdoc.org/
* https://www.linux.com/training-tutorials/linux-documentation-project/
* https://linux.die.net/man/7/ldp
* https://docs.kernel.org/


