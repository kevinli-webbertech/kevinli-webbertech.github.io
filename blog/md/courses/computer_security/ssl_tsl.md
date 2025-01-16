
# EC2 instance webbertech.com

## pub/private + pem file


```
ssh -i "webbertech.com.pem" ubuntu@ec2-3-19-83-70.us-east-2.compute.amazonaws.com

MacBookPro:deployment kevinli$ ssh -i "webbertech.com.pem" ubuntu@ec2-3-19-83-70.us-east-2.compute.amazonaws.com
The authenticity of host 'ec2-3-19-83-70.us-east-2.compute.amazonaws.com (3.19.83.70)' can't be established.
ED25519 key fingerprint is SHA256:aDTbMRIYg3tBUVNj2eGLcVgxyno9b+JZ1lpjOjQTkgE.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'ec2-3-19-83-70.us-east-2.compute.amazonaws.com' (ED25519) to the list of known hosts.
Welcome to Ubuntu 24.04 LTS (GNU/Linux 6.8.0-1012-aws x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

 System information as of Thu Jan 16 03:58:56 UTC 2025

  System load:  0.0               Processes:             107
  Usage of /:   45.3% of 6.71GB   Users logged in:       0
  Memory usage: 53%               IPv4 address for enX0: 172.31.6.147
  Swap usage:   0%

 * Ubuntu Pro delivers the most comprehensive open source security and
   compliance features.

   https://ubuntu.com/aws/pro

Expanded Security Maintenance for Applications is not enabled.

142 updates can be applied immediately.
To see these additional updates run: apt list --upgradable

Enable ESM Apps to receive additional future security updates.
See https://ubuntu.com/esm or run: sudo pro status


*** System restart required ***
Last login: Fri Sep 13 22:22:48 2024 from 3.16.146.4
```

And in the `known_host`, the following entry is added,

```shell
ec2-3-19-83-70.us-east-2.compute.amazonaws.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAHVy2/B0tcRMumfnrIGv7mQ0wL//ZorTBab0ulGsvxq
ec2-3-19-83-70.us-east-2.compute.amazonaws.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDL4CSC9ACxgnVJdFXbBqPawW/vIKxC0SVEpl+yZM6IqhqKqvR9a9D6/6wCN4dlTSGdVN9Gka52zl8Mm8QcMZTWeEEEhRwEW7v4qMtfpkmasBEpfMXG7gMwolGLpF9sT+kpQkV4G5bqjVNGUwnWPOvPc+/sX7FmLp7eNyaN5uudLu5PLKWtD8dbuTcOvpuMtqhTr9OvdLso+Y8cNLREzqgWQS37qt+a2TVMD+KqABATz+bYoei9BN63B0q3zNz/gMTR/8s5yKYJjO3/nUcyn27rjRwaxmkE5rxPuAJ67TNgq5dfzj5eO/BQjOzCy49H7fHLCkm+DTnLXUPy7c4rhjZr/XHMSlA+Ai7hQXqDQmLwcDt0PUDRHcEZdcqmVcUwKbDlrknnEqkOnoncLUqUyHiVOjveZe9AGEKgIysQI5oxnFP67j0PiXuz5TCSTNWQvrYkp65+RRVvYAef1RQAM62ln2PuZDpMC523i3fwSAfXPd5005q+G0IO9DbCymU2uZE=
ec2-3-19-83-70.us-east-2.compute.amazonaws.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBGDUbkUPuS8xoCI/9hmN25VWUmsD6o0SX21klom9dq7dzkx7ewxGkMcoZfJ8ZIHKtQL3mO72yPEMBprgJZ3Mmw4=
```

After login to the EC2 instance, in its `.ssh/authorized_keys`,

```shell
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCk0+8nlZBfWHIHLm1WrDpYBILH0jTeD5LOjyrk8Se8vZqIuqDwW5KAVRgIFcALmn9uatIoMDuMS+tv2UNdH8sWr3VS6diij23yQu9iMWRUfwTeymS687Z1JUjFwZg31GGudm38m4Xu1qm20HB/cQMlVOakc26CVMH+k4WQo4K5w1FIZuHzhC/8FI+za7i9Mth8Eg/PYhN160+oGdsLae1gEgUo8f28Ic795gBKFc76UH0LYEZ4vpyJg799QNbSLN4Ld+wHCDy9S7sT8G9rSazKSXqHru2TMn1pRNUJ0CeoS5QwAfMllkoLMsDNMFhR/FZDzbHbnLvtyooWvFC7L1U9 webbertech.com
```

## EC2 insance kltutor.com

`ssh -i "kltutor.com.pem" ubuntu@ec2-18-221-239-84.us-east-2.compute.amazonaws.com`

In the client's `known_hosts`, we see the following,

```
ec2-3-19-83-70.us-east-2.compute.amazonaws.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAHVy2/B0tcRMumfnrIGv7mQ0wL//ZorTBab0ulGsvxq
ec2-3-19-83-70.us-east-2.compute.amazonaws.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDL4CSC9ACxgnVJdFXbBqPawW/vIKxC0SVEpl+yZM6IqhqKqvR9a9D6/6wCN4dlTSGdVN9Gka52zl8Mm8QcMZTWeEEEhRwEW7v4qMtfpkmasBEpfMXG7gMwolGLpF9sT+kpQkV4G5bqjVNGUwnWPOvPc+/sX7FmLp7eNyaN5uudLu5PLKWtD8dbuTcOvpuMtqhTr9OvdLso+Y8cNLREzqgWQS37qt+a2TVMD+KqABATz+bYoei9BN63B0q3zNz/gMTR/8s5yKYJjO3/nUcyn27rjRwaxmkE5rxPuAJ67TNgq5dfzj5eO/BQjOzCy49H7fHLCkm+DTnLXUPy7c4rhjZr/XHMSlA+Ai7hQXqDQmLwcDt0PUDRHcEZdcqmVcUwKbDlrknnEqkOnoncLUqUyHiVOjveZe9AGEKgIysQI5oxnFP67j0PiXuz5TCSTNWQvrYkp65+RRVvYAef1RQAM62ln2PuZDpMC523i3fwSAfXPd5005q+G0IO9DbCymU2uZE=
ec2-3-19-83-70.us-east-2.compute.amazonaws.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBGDUbkUPuS8xoCI/9hmN25VWUmsD6o0SX21klom9dq7dzkx7ewxGkMcoZfJ8ZIHKtQL3mO72yPEMBprgJZ3Mmw4=
```

In the EC2's `.ssh/authorized_keys`, we can see the following,

```shell
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCm9EObK0HOjG7DawRE0tJmVtYQzthpZqTKu7k9mlfX6xcQVUFUO1C9K+pPRZ9D63R06xiXMAYHkozGXbXWG6mT4MoAeZr7NjGBQb4v//VAy2fTX9utLVrQkXzWL0JegywtvGtgzHESbUL5wVIYP2shaHaCNHv2AMxKZL9txpUfyL2iUibZJkLyHW1rM5adlb51InQiFcw8cvy2fxrQGfVVPlgPKJIy9CftuPc4i1RWqgdruckfNjWc0CMgfWd/e6xGI8UMbsKqAyCaO5hGQZAa+zlLvoerpkcDkJ9WqO3q7pTbYeXcxvGK3+IZNBbBN0gNdBfV5KYaeadW3yGefynb kltutor.com
```