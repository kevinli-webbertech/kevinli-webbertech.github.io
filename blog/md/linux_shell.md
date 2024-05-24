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

## Files Operations
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

#### cut examples

```
touch words.txt

happy
world
hello John

[img missing]
List without ranges:
cut -b 1,2,3 state.txt

[img missing]

cut -b 1-3,5-7 state.txt

[img missing]

```

Special Form: Selecting bytes from beginning to end of line
In this, 1- indicate from 1st byte to end byte of a line

`cut -b 1- state.txt`

In this, -3 indicate from 1st byte to 3rd byte of a line

`cut -b -3 state.txt`

Cut by Character (-c) Using cut Command

-c (column): To cut by character use the -c option. This selects the characters given to the -c option. This can be a list of numbers separated comma or a range of numbers separated by hyphen(-).

Tabs and backspaces are treated as a character. It is necessary to specify list of character numbers otherwise it gives error with this option.

Syntax:

`cut -c [(k)-(n)/(k),(n)/(n)] filename`

Extract specific characters:

`cut -c 2,5,7 state.txt`

Extract first seven characters:

`cut -c 1-7 state.txt`

Above command prints starting from first character to end.

`cut -c -5 state.txt`

Above command prints starting position to the fifth character. 

Cut by Field (-f) Using cut Command

-f (field): -c option is useful for fixed-length lines. Most unix files doesn’t have fixed-length lines. To extract the useful information you need to cut by fields rather than columns. List of the fields number specified must be separated by comma. Ranges are not described with -f option. cut uses tab as a default field delimiter but can also work with other delimiter by using -d option.


`cut -f 1 state.txt`

If `-d` option is used then it considered space as a field separator or delimiter:

`cut -d " " -f 1 state.txt`

Extract fields 1 to 4:
Command prints field from first to fourth of each line from the file.

`cut -d " " -f 1-4 state.txt`


ref: https://www.geeksforgeeks.org/cut-command-linux-examples/

#### tr examples

The tr command in Linux is used for translating or deleting characters. It reads from standard input and writes to standard output. Here are some common examples of how to use the tr command:

Basic Usage
Translate Lowercase to Uppercase

sh
Copy code
echo "hello world" | tr 'a-z' 'A-Z'
Output:

Copy code
HELLO WORLD
Translate Uppercase to Lowercase

sh
Copy code
echo "HELLO WORLD" | tr 'A-Z' 'a-z'
Output:

Copy code
hello world
Character Deletion
Delete Specific Characters

sh
Copy code
echo "hello 123 world" | tr -d '0-9'
Output:

Copy code
hello  world
Delete All Vowels

sh
Copy code
echo "hello world" | tr -d 'aeiou'
Output:

Copy code
hll wrld
Character Complement (Negation)
Delete All Characters Except Digits
sh
Copy code
echo "hello 123 world" | tr -cd '0-9'
Output:
Copy code
123
Squeeze Repeated Characters
Squeeze Repeated Spaces

sh
Copy code
echo "hello   world" | tr -s ' '
Output:

Copy code
hello world
Squeeze Repeated Characters

sh
Copy code
echo "hellooooo   wooorld" | tr -s 'o'
Output:

Copy code
helloo   woorld
Character Ranges and Classes
Replace Non-Alphanumeric Characters with Newlines

sh
Copy code
echo "hello, world! 123" | tr -cs '[:alnum:]' '\n'
Output:

Copy code
hello
world
123
Remove All Non-Alphanumeric Characters

sh
Copy code
echo "hello, world! 123" | tr -cd '[:alnum:]'
Output:

Copy code
helloworld123
Use with Other Commands
Convert Newlines to Spaces

sh
Copy code
cat file.txt | tr '\n' ' '
Sort and Remove Duplicates from a List

sh
Copy code
echo -e "banana\napple\norange\napple\nbanana" | tr '\n' '\0' | xargs -0 -n1 | sort | uniq
Translating Characters
ROT13 Encryption
sh
Copy code
echo "hello" | tr 'A-Za-z' 'N-ZA-Mn-za-m'
Output:
Copy code
uryyb
Advanced Usage
Change Case and Remove Non-Alphanumeric Characters
sh
Copy code
echo "Hello, World! 123" | tr 'A-Z' 'a-z' | tr -cd '[:alnum:] \n'
Output:
Copy code
hello world 123
Usage in Scripts
Remove Spaces from a String (in a script)
sh
Copy code
STRING="hello world"
CLEANED_STRING=$(echo "$STRING" | tr -d ' ')
echo $CLEANED_STRING
Output:
Copy code
helloworld
Combining with Other Utilities
Create a Filename-Friendly String
sh
Copy code
echo "File Name: With Special/Characters!" | tr -cd '[:alnum:]._-'
Output:
Copy code
FileNameWithSpecialCharacters
These examples cover some of the most common and useful applications of the tr command in Linux. Adjust the arguments and inputs to fit specific needs in your scripts or command line operations.







## Text find, replace and regex

* `sed`	Finds, replaces, or deletes patterns in a file
* `awk`	Finds and manipulates patterns in a file
* `grep`	Searches a string within a file

## Package and software management (debian)

* `apt`
* `apt-get`	Manages Debian-based distros package libraries
* `dpkg`    dpkg is a medium-level tool to install, build, remove and manage Debian packages.

### apt and apt-get examples

The apt command line tool provides a higher-level user interface for end users with intuitive commands, resulting behaviors, and security features. In contrast, the command apt-get is a low-level interface that communicates more closely with core Linux processes. The apt command is a more user-friendly package manager than apt-get.

Used in Linux-based operating systems to update the package lists for available software packages from the configured repositories.

`sudo apt-get upgrade`

This command is used to install the latest versions of the packages currently installed on the user’s system from the sources enumerated in /etc/apt/sources.list. The installed packages which have new packages available are retrieved and installed. You need to perform an update before the upgrade so that apt-get knows that new versions of packages are available.

`sudo apt-get install [package_name]`

This command is used to install or upgrade packages. 


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
* `lsof` List All Open Files 
* `which` Find file location which is on the path
* `whomai` Currently logged-in user
* `uptime` Time system has been up since last reboot
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
```

#### `lsof usage`

FD – stands for a File descriptor and may see some of the values as:

* cwd current working directory
* rtd root directory
* txt program text (code and data)
* mem memory-mapped file

Also in FD column numbers like 1u is actual file descriptor and followed by u,r,w of its mode as:

`r` for read access.
`w` for write access.
`u` for read and write access.
`TYPE` – of files and it’s identification.

`DIR` – Directory
`REG` – Regular file
`CHR` – Character special file.
`FIFO` – First In First Out


```
xiaofengli@xiaofenglx:~/code/codebank$ lsof| grep -i 8080
apache2   8072 8080 apache2           www-data  cwd   unknown                                         /proc/8072/task/8080/cwd (readlink: Permission denied)
apache2   8072 8080 apache2           www-data  rtd   unknown                                         /proc/8072/task/8080/root (readlink: Permission denied)
apache2   8072 8080 apache2           www-data  txt   unknown                                         /proc/8072/task/8080/exe (readlink: Permission denied)
apache2   8072 8080 apache2           www-data NOFD                                                   /proc/8072/task/8080/fd (opendir: Permission denied)

```

List User Specific Opened Files

`lsof -u xiaofengli`

Find Processes Running on Specific Port
`lsof -i TCP:22`

List Only IPv4 & IPv6 Open Files

`lsof -i 4`

`lsof -i 6`

List Open Files of TCP Port Ranges 1-1024

` lsof -i TCP:1-1024`

Exclude User with ‘^’ Character

`lsof -i -u^root`

Find Out who’s Looking What Files and Commands?

`lsof -i -u xiaofenglx`

List all Network Connections

The following command with option ‘-i’ shows the list of all network connections ‘LISTENING & ESTABLISHED’.

`lsof -i`

Search by PID

`lsof -p 1`

Kill all Activity of Particular User

```
kill -9 `lsof -t -u tecmint`
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
* `ssh`

### Examples

Get all the links recursively. 

`wget -r https://docs.python.org/3/tutorial/index.html`

`curl` to send http post.

```
curl --location '192.168.1.186:9200/school*/_search' \
--header 'Content-Type: application/json' \
--data '{
   "query":{
       "match" : {
         "rating":"4.5"
      }
   }
}'
```

`curl` to do http get.
```
curl --location --request GET '192.168.1.186:9401/schools/_search' \
--header 'Content-Type: application/json' \
--data '{
   "query":{
      "match_all":{}
   }
}'
```

`netstat usage`

- -a To list all listening ports, using both TCP and UDP, use `netstat -a`
- -t only tcp ports connections
- -u only udp ports connections
- -l To list all actively listening ports (both TCP and UDP)
- -s To pull and view network statistics sorted by protocol
- -p show pid
- -r The -r option of netstat displays the IP routing table
- -i To view send/receive stats by interface

Raw network stats,

`netstat --statistics --raw`



```
ssh tunnel
```

## System commands

### `compgen`

compgen -c will list all the commands you could run.
compgen -a will list all the aliases you could run.
compgen -b will list all the built-ins you could run.
compgen -k will list all the keywords you could run.
compgen -A function will list all the functions you could run.
compgen -A function -abck will list all the above in one go.

### `systemctl`

`Systemd` is a system and service manager for Linux; a drop-in replacement for the init process, which is compatible with SysV and LSB init scripts, and the `systemctl` command is the primary tool to manage systemd.

**Example**

`# systemctl list-units --type=service`

OR

`# systemctl --type=service`

```
# systemctl list-units --type=service --state=active
OR
# systemctl --type=service --state=active

```

https://www.tecmint.com/list-all-running-services-under-systemd-in-linux/


## Firewall & IPtable
https://www.ninjaone.com/blog/how-to-configure-a-linux-firewall/#:~:text=After%20you%20configure%20a%20Linux,traffic%20based%20on%20predefined%20rules.

## File and file descriptor


## Shell scripting


### function

### loop

### case

### redirect


## Linux Desktop Environment

### GNOME
GNOME is used both by Fedora and Ubuntu.
Nautilus offers a simple and integrated way of managing files and browsing the file system.

- Activities overview
- GNOME Software Center
- GNOME Commander: a free open source graphic file manager for linux desktop.
- GNOME Display Manager (GDM): A display manager (a graphical login manager) for the windowing systems X11 and Wayland
- Mutter:A portmanteau of "Metacity" and "Clutter", Mutter can function as a standalone window manager for GNOME-like desktops

#### Nautilus

Files, also known as Nautilus, is the default file manager of the GNOME desktop. It provides a simple and integrated way of managing your files and browsing your file system.

Nautilus supports all the basic functions of a file manager and more. It can search and manage your files and folders, both locally and on a network, read and write data to and from removable media, run scripts, and launch apps. It has three views: Icon Grid, Icon List, and Tree List. Its functions can be extended with plugins and scripts.

More information can be found here,

https://apps.gnome.org/Nautilus/#:~:text=Files%2C%20also%20known%20as%20Nautilus,a%20file%20manager%20and%20more

### GDM

GNOME Display Manager (GDM) is a display manager (a graphical login manager) for the windowing systems X11 and Wayland. GDM was written from scratch and does not contain any XDM or X Consortium code.

The X Window System by default uses the XDM display manager. However, resolving XDM configuration issues typically involves editing a configuration file. GDM allows users to customize or troubleshoot settings without having to resort to a command line. 

GDM comprises the following components:

- Chooser – a program used to select a remote host as the source for a remote display on the attached display (gdm-host-chooser)
- Greeter – the graphical login window (provided by GNOME Shell)
- Pluggable authentication module (PAM)
- X Display Manager Control Protocol (XDMCP)

#### PAM

A pluggable authentication module (PAM) is a mechanism to integrate multiple low-level authentication schemes into a high-level application programming interface (API). 

PAM allows programs that rely on authentication to be written independently of the underlying authentication scheme.

![pam architecture](https://kevinli-webbertech.github.io/blog/images/linux/pam.png)

History,

* 1995: It was first proposed by Sun Microsystems in an Open Software Foundation Request for Comments (RFC) 86.0 dated October 1995.

* 1996: It was adopted as the authentication framework of the Common Desktop Environment. As a stand-alone open-source infrastructure, PAM first appeared in Red Hat Linux 3.0.4 in August 1996 in the Linux PAM project. 

* X/Open Single Sign-on (XSSO) 
Since no central standard of PAM behavior exists, there was a later attempt to standardize PAM as part of the X/Open UNIX standardization process, resulting in the X/Open Single Sign-on (XSSO) standard.

* OpenPAM
This standard was not ratified, but the standard draft has served as a reference point for later PAM implementations (for example, OpenPAM).

#### X Display Manager Control Protocol (XDMCP)

* In X window System, the X server runs.

* The X window display manager offers a graphics login manager to start a login session to ask users for username and password.

* The X window display manager acts to do the same functionality as the getty and login on character-mode terminals.

* The X window display manager can connect remotely, then it acts like a telnet server.

* The XDM (the X Window Display Manager) originated in X11R3. 
It only reads from Xservers file thus every time a user switched a terminal off and on, it will not know. In X11R4, with XDMCP, the X server must actively request a display manager connection from the host. An X server using XDMCP therefore no longer requires an entry in Xservers.

1988: X11 Release 3 introduced display managers in October 1988 with the aim of supporting the standalone X terminals, just coming onto the market. 

1989: X11R4 introduced the X Display Manager Control Protocol (XDMCP) in December 1989 to fix problems in the X11R3 implementation.

* The X Display Manager Control Protocol (XDMCP) uses UDP port 177.
And it keeps a secret key and have a hand-shake package with the X server to authenticate itself. When the session is established,
During the session, the server can send KeepAlive packets to the display manager at intervals.  If the display manager fails to respond with an Alive packet within a certain time, the X server presumes that the display manager has ceased running, and can terminate the connection.

* XDMCP security concern. 

One problem with XDMCP is that, similarly to telnet, the authentication takes place unencrypted.

#### ref

- https://www.gnome.org/
- https://apps.gnome.org/Nautilus/#:~:text=Files%2C%20also%20known%20as%20Nautilus,a%20file%20manager%20and%20more.
- https://extensions.gnome.org/
- https://en.wikipedia.org/wiki/Pluggable_authentication_module

### KDE

KDE is an international free software community that develops free and open-source software. As a central development hub, it provides tools and resources that allow collaborative work on this kind of software. Well-known products include the Plasma Desktop, KDE Frameworks, and a range of cross-platform applications such as Amarok, digiKam, and Krita that are designed to run on Unix and Unix-like operating systems, Microsoft Windows, and Android.

KDE was founded in 1996 by Matthias Ettrich, a student at the University of Tübingen.
In the beginning Matthias Ettrich chose to use Trolltech's Qt framework for the KDE project. Other programmers quickly started developing KDE/Qt applications, and by early 1997, a few applications were being released. On 12 July 1998 the first version of the desktop environment, called KDE 1.0, was released. The original GPL licensed version of this toolkit only existed for platforms which used the X11 display server, but with the release of Qt 4, LGPL licensed versions are available for more platforms. This allowed KDE software based on Qt 4 or newer versions to theoretically be distributed to Microsoft Windows and OS X.

The KDE Marketing Team announced a rebranding of the KDE project components on 24 November 2009. Motivated by the perceived shift in objectives, the rebranding focused on emphasizing both the community of software creators and the various tools supplied by the KDE, rather than just the desktop environment.

What was previously known as KDE 4 was split into KDE Plasma Workspaces, KDE Applications, and KDE Platform (now KDE Frameworks) bundled as KDE Software Compilation 4. Since 2009, the name KDE no longer stands for K Desktop Environment, but for the community that produces the software.

#### ref

- https://en.wikipedia.org/wiki/KDE
- https://kde.org/
- https://kdeconnect.kde.org/
- https://neon.kde.org/
- https://kde.org/plasma-desktop/

### xfce

#### ref
- https://www.vpsserver.com/gnome-vs-xfce-vs-kde/#:~:text=KDE%2C%20GNOME%2C%20and%20Xfce%20are,of%20the%20Unix%20operating%20system.

- https://1gbits.com/blog/best-linux-desktop-environment/#:~:text=The%20most%20well%2Dknown%20Linux,be%20user%2Dfriendly%20and%20adaptable.

- https://www.vpsserver.com/gnome-vs-xfce-vs-kde/#:~:text=Xfce%20is%20faster%20than%20KDE,quicker%20desktop%20environment%20than%20GNOME.

- https://cloudzy.com/blog/linux-mint-cinnamon-vs-mate-vs-xfce/#:~:text=Unlike%20Linux%20Mint%20Cinnamon%20and,but%20has%20its%20own%20libraries.

- https://alternativeto.net/software/gnome/

## Common XDM

* GDM, GNOME implemation
* KDM, KDE implementation
* SDDM, KDE Plasma 5 and LXQt. Successor to KDM.
* LightDM, a lightweight, modular, cross-desktop, fully themeable desktop display manager by Canonical Ltd.
* TWin, the TDE window manager
* dtlogin (shipped with CDE)
* xlogin display manager, a lightweight, secure and login like console display manager for X, written in C.

Note: `Source coming from Wikipedia`

#### ref
- https://en.wikipedia.org/wiki/X_display_manager#XDMCP

## X Server

The X Window System (X11, or simply X) is a windowing system for bitmap displays, common on Unix-like operating systems.

X originated as part of Project Athena at Massachusetts Institute of Technology (MIT) in 1984. The X protocol has been at version 11 (hence "X11") since September 1987. The X.Org Foundation leads the X project, with the current reference implementation, X.Org Server, available as free and open-source software under the MIT License and similar permissive licenses.

![x server](https://kevinli-webbertech.github.io/blog/images/linux/xserver.png)

#### ref

- https://www.x.org/releases/X11R7.6/doc/man/man1/Xserver.1.xhtml

## GTK

GTK (formerly GIMP ToolKit and GTK+) is a free software cross-platform widget toolkit for creating graphical user interfaces (GUIs). It is licensed under the terms of the GNU Lesser General Public License, allowing both free and proprietary software to use it. It is one of the most popular toolkits for the Wayland and X11 windowing systems.

The GTK team releases new versions on a regular basis. GTK 4 and GTK 3 are maintained, while GTK 2 is end-of-life. GTK1 is independently maintained by the CinePaint project. GTK 4 dropped the + from the name.

* GTK Drawing Kit (GDK): GDK acts as a wrapper around the low-level functions provided by the underlying windowing and graphics systems.

#### ref
- https://www.gtk.org/
- DevConf.cz

## Wayland

Wayland is a communication protocol that specifies the communication between a display server and its clients, as well as a C library implementation of that protocol. A display server using the Wayland protocol is called a Wayland compositor, because it additionally performs the task of a compositing window manager.

Wayland is developed by a group of volunteers initially led by Kristian Høgsberg as a free and open-source community-driven project with the aim of replacing the X Window System with a secure and simpler windowing system for Linux and other Unix-like operating systems. The project's source code is published under the terms of the MIT License, a permissive free software licence.

As part of its efforts, the Wayland project also develops a reference implementation of a Wayland compositor called Weston.

#### ref


### Check the XDM

- Display Environment Variable: `echo $DISPLAY` and `echo $XDG_SESSION_TYPE`
- Check process: `ps aux | grep [X]` or `ps aux | grep [w]ayland`
- Using loginctl Command: `$ loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p Type`

## Over all Linux References and Websites

* https://man7.org/tlpi/index.html
* https://www.dpkg.org/
* https://manpages.ubuntu.com/manpages/trusty/man1/dpkg.1.html
* https://tldp.org/
* https://github.com/tLDP
* https://www.linuxdoc.org/
* https://www.linux.com/training-tutorials/linux-documentation-project/
* https://linux.die.net/man/7/ldp
* https://docs.kernel.org/


