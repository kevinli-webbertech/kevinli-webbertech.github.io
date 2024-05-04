# Linux Ref

## Linux file system commands

* ls	Lists a directory’s content
* pwd	Shows the current working directory’s path
* cd	Changes the working directory
* mkdir	Creates a new directory
* rm	Deletes a file
* cp	Copies files and directories, including their content
* mv	Moves or renames files and directories
* touch	Creates a new empty file
* file	Checks a file’s type
* zip and unzip	Creates and extracts a ZIP archive
* tar	Archives files without compression in a TAR format
* nano, vi, and jed	Edits a file with a text editor
* history	Lists previously run commands
* man	Shows a command’s manual
* echo	Prints a message as a standard output
* ln	Links files or directories
* alias and unalias	Sets and removes an alias for a file or command
* date Display system time
* cal	Displays a calendar in Terminal

## Permissions

* sudo	Runs a command as a superuser
* su	Runs programs in the current shell as another user
* chmod	Modifies a file’s read, write, and execute permissions
* chown	Changes a file, directory, or symbolic link’s ownership
* useradd and userdel	Creates and removes a user account

# Files Operations
* cat	Lists, combines, and writes a file’s content as a standard output
* less  Read file
* more  Read file
* head	Displays a file’s first ten lines
* tail	Prints a file’s last ten lines
* sort	Reorders a file’s content
* cut	Sections and prints lines from a file
* diff	Compares two files’ content and their differences
* grep	Searches a string within a file
* sed	Finds, replaces, or deletes patterns in a file
* awk	Finds and manipulates patterns in a file
* tee	Prints command outputs in Terminal and a file
* locate	Finds files in a system’s database
* find	Outputs a file or folder’s location
* wc    counting utility

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

## Package and software management (debian)

* apt
* apt-get	Manages Debian-based distros package libraries
* dpkg      Package management

### apt and apt-get

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

## System Information

* df	Displays the system’s overall disk space usage
* du	Checks a file or directory’s storage consumption
* top	Displays running processes and the system’s resource usage
* htop	Works like top but with an interactive user interface
* ps	Creates a snapshot of all running processes
* uname	Prints information about your machine’s kernel, name, and hardware
* hostname	Shows your system’s hostname
* time	Calculates commands’ execution time
* systemctl	Manages system services
* watch	Runs another command continuously
* jobs	Displays a shell’s running processes with their statuses
* kill	Terminates a running process
* shutdown	Turns off or restarts the system
* bg
* fg

## Network

* ping	Checks the system’s network connectivity
* wget	Downloads files from a URL
* curl	Transmits data between servers using URLs
* scp	Securely copies files or directories to another system
* rsync	Synchronizes content between directories or machines
* lfconfig	Displays the system’s network interfaces and their configurations
* netstat	Shows the system’s network information, like routing and sockets
* traceroute	Tracks a packet’s hops to its destination
* nslookup	Queries a domain’s IP address and vice versa
* dig	Displays DNS information, including record types



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


