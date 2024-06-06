# Linux Ref

## Linux file system commands

* `man` Shows a command’s manual
* `echo` Prints a message as a standard output
* `ls` Lists a directory’s content
* `pwd` Shows the current working directory’s path
* `cd` Changes the working directory
* `mkdir` Creates a new directory
* `rm` Deletes a file
* `cp` Copies files and directories, including their content
* `mv` Moves or renames files and directories
* `touch` Creates a new empty file
* `file` Checks a file’s type
* `zip` and unzip Creates and extracts a ZIP archive
* `tar` Archives files without compression in a TAR format
* `nano`, vi, and jed Edits a file with a text editor
* `history` Lists previously run commands
* `date` Display system time
* `cal` Displays a calendar in Terminal. (not builtin, need to install `ncal`)
* `calc` Calculator
* `tree` folder structure

## VIM shortcuts [TODO]

## Profile, Configuration and Path

* `ln` Links files or directories
* `alias` and `unalias` Sets and removes an alias for a file or command
* `export` export definition of system variable
* `source` execute the system profile

## Permissions

* `sudo` Runs a command as a superuser
* `su` Runs programs in the current shell as another user
* `chmod` Modifies a file’s read, write, and execute permissions
* `chown` Changes a file, directory, or symbolic link’s ownership
* `useradd` and `userdel` Creates and removes a user account

## Files Operations

* `cat` Lists, combines, and writes a file’s content as a standard output
* `less` Read file
* `more`  Read file
* `head` Displays a file’s first ten lines
* `tail` Prints a file’s last ten lines
* `sort` Reorders a file’s content
* `cut` Sections and prints lines from a file
* `diff` Compares two files’ content and their differences
* `tee` Prints command outputs in Terminal and a file
* `locate` Finds files in a system’s database
* `find` Outputs a file or folder’s location
* `wc`   counting utility
* `tr` ??
* `rev` ??

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

ref: <https://www.geeksforgeeks.org/cut-command-linux-examples/>

#### sort examples

Sure, here's a simple example of using the `sort` command in Linux:

Let's say you have a file called `names.txt` with the following content:

```
John
Alice
Bob
Zoe
```

To sort the names alphabetically, you can use the `sort` command like this:

```bash
sort names.txt
```

This will output:

```
Alice
Bob
John
Zoe
```

By default, `sort` sorts lines of text in ascending order, which means alphabetically for strings. If you want to sort in descending order, you can use the `-r` option:

```
sort -r names.txt
```

This will output:

```
Zoe
John
Bob
Alice
```

You can also use `sort` to sort the output of a command. For example, if you have a list of numbers stored in a file called `numbers.txt`, you can sort them like this:

```
cat numbers.txt | sort -n
```

This will sort the numbers numerically in ascending order. If you want to sort them in descending order, you can use:

```
cat numbers.txt | sort -n -r
```

These are just a few examples of how you can use the `sort` command in Linux to sort text data. There are many more options available, so you may want to refer to the `sort` manual (`man sort`) for more information.

## Text find, replace and regex

* `sed` Finds, replaces, or deletes patterns in a file
* `awk` Finds and manipulates patterns in a file
* `grep` Searches a string within a file

## Package and software management (debian)

* `apt`
* `apt-get` Manages Debian-based distros package libraries
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

* `df` Displays the system’s overall disk space usage
* `du` Checks a file or directory’s storage consumption
* `top` Displays running processes and the system’s resource usage
* `vmstat`  Report virtual memory statistic
* `htop` Works like top but with an interactive user interface (not default command)
* `ps` Creates a snapshot of all running processes
* `uname` Prints information about your machine’s kernel, name, and hardware
* `hostname` Shows your system’s hostname
* `time` Calculates commands’ execution time
* `systemctl` Manages system services
* `watch` Runs another command continuously
* `jobs` Displays a shell’s running processes with their statuses
* `kill` Terminates a running process
* `bg` ??
* `fg` ??
* `lsof` List All Open Files
* `which` Find file location which is on the path
* `whomai` Currently logged-in user
* `uptime` Time system has been up since last reboot
* `shutdown` Turns off or restarts the system
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
1 ./BloggerDev
1 ./.git/branches
1 ./.git/refs/heads
1 ./.git/refs/tags
1 ./.git/refs/remotes/origin
1 ./.git/refs/remotes
1 ./.git/refs
1 ./.git/info
2 ./.git/objects/97

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

`lsof -i TCP:1-1024`

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

* `ping` Checks the system’s network connectivity
* `wget` Downloads files from a URL
* `curl` Transmits data between servers using URLs
* `scp` Securely copies files or directories to another system
* `rsync` Synchronizes content between directories or machines
* `lfconfig` Displays the system’s network interfaces and their configurations
* `netstat` Shows the system’s network information, like routing and sockets
* `traceroute` Tracks a packet’s hops to its destination
* `nslookup` Queries a domain’s IP address and vice versa
* `dig` Displays DNS information, including record types
* `ssh`

### Examples of weget

Get all the links recursively.

`wget -r https://docs.python.org/3/tutorial/index.html`

### Examples of curl

-X, ??
-L, ??
-d, ??
-o, ??

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

#### netstat usage

* -a To list all listening ports, using both TCP and UDP, use `netstat -a`
* -t only tcp ports connections
* -u only udp ports connections
* -l To list all actively listening ports (both TCP and UDP)
* -s To pull and view network statistics sorted by protocol
* -p show pid
* -r The -r option of netstat displays the IP routing table
* -i To view send/receive stats by interface

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

<https://www.tecmint.com/list-all-running-services-under-systemd-in-linux/>

## Firewall & IPtable

<https://www.ninjaone.com/blog/how-to-configure-a-linux-firewall/#:~:text=After%20you%20configure%20a%20Linux,traffic%20based%20on%20predefined%20rules>.

## File and file descriptor

## Over all Linux References and Websites

* <https://man7.org/tlpi/index.html>
* <https://www.dpkg.org/>
* <https://manpages.ubuntu.com/manpages/trusty/man1/dpkg.1.html>
* <https://tldp.org/>
* <https://github.com/tLDP>
* <https://www.linuxdoc.org/>
* <https://www.linux.com/training-tutorials/linux-documentation-project/>
* <https://linux.die.net/man/7/ldp>
* <https://docs.kernel.org/>
