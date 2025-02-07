sudo
The sudo command allows users listed in /etc/sudoers to run administrative commands. You can configure /etc/sudoers to set limits on the root privileges granted to a specific user.

To use sudo commands you don't need to give root password. A user with appropriate right from /etc/sudoers can execute root privilege command form his own passwords.

Red Hat Enterprise Linux provides some features that make working as root somewhat safer. For example, logins using the ftp and telnet commands to remote computers are disabled by default.

Limiting Access to sudo
You can limit access to the sudo command. Regular users who are authorized in /etc/sudoers can access administrative commands with their own password. You don't need to give out the administrative password to everyone who thinks they know as much as you do about Linux. To access /etc/sudoers in the vi editor, run the visudo command.

linux vi /etc/sudoers

From the following directive, the root user is allowed full access to administrative commands:


For example, if you want to allow user vinita full administrative access, add the following directive to /etc/sudoers:

 root ALL=(ALL) ALL vinita ALL=(ALL) ALL 
In this case, all vinita needs to do to run an administrative command such as starting the network service from her regular account is to run the following command, entering her own user password (note the regular user prompt, $):

 $ sudo /sbin/service network restart Password: 


You can even allow special users administrative access without a password. As suggested by the comments, the following directive in /etc/sudoers would allow all users in the wheel group to run administrative commands without a password:

 %wheel ALL=(ALL) NOPASSWD: ALL 
But you don't have to allow full administrative access. For example, if you want to allow those in the %users group to shut down the local system, you can activate the following directive:

 %users localhost=/sbin/shutdown -h now 

 ## Permission part added here

 
 ## Ref

 https://www.computernetworkingnotes.com/linux-tutorials/linux-user-profile-management-and-environment-variable.html#google_vignette