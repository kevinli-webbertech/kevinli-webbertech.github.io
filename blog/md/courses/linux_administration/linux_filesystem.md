# **Linux File Systems Overview**

Linux supports multiple file systems, each designed for different use cases. Here are some common ones:

| **File System** | **Description** | **Use Case** |
|---------------|---------------|------------|
| **ext4** | Default Linux file system with journaling and large file support. | General-purpose use (desktops, servers). |
| **XFS** | High-performance journaling file system, scalable. | Large-scale storage, enterprise servers. |
| **Btrfs** | Advanced features like snapshots and checksumming. | Modern storage solutions, data integrity. |
| **ZFS** | RAID, snapshots, and compression built-in. | Enterprise storage, data protection. |
| **F2FS** | Optimized for NAND flash storage (SSDs). | Mobile devices, SSD-based systems. |
| **NTFS** | Microsoft file system with journaling. | Used for Windows partitions in Linux. |
| **VFAT/exFAT** | Microsoft-compatible file systems for removable media. | USB drives, SD cards. |

## **Check and Manage File Systems**

- **List available file systems**:  
  ```bash
  cat /proc/filesystems
  ```

- **Check file system type of a partition**:  
  ```bash
  lsblk -f
  ```

- **Format a partition with ext4**:  
  ```bash
  sudo mkfs.ext4 /dev/sdX
  ```

- **Mount a file system**:  
  ```bash
  sudo mount /dev/sdX /mnt
  ```

- **Check disk usage**:  
  ```bash
  df -h
  ```

## **Automounting File Systems in Linux**

To automatically mount a file system at boot, follow these steps:

### **1. Find the Disk UUID**

```bash
blkid
```

- Note the UUID of the partition you want to mount.

### **2. Edit `/etc/fstab`**

```bash
sudo nano /etc/fstab
```

- Add an entry like this:

  ```
  UUID=your-uuid /mnt/your-mount-point ext4 defaults 0 2
  ```

  *(Replace `your-uuid` with the actual UUID and `/mnt/your-mount-point` with your desired mount location.)*

### **3. Create the Mount Point**

```bash
sudo mkdir -p /mnt/your-mount-point
```

### **4. Test the Mount**

```bash
sudo mount -a
```

- If no errors appear, the setup is correct.

Now, the partition will mount automatically at startup!

## Appendix - My mount in my linux

```shell
xiaofengli@xiaofenglx:~$ mount |grep ntfs
/dev/sdb1 on /mnt/ntfs type fuseblk (rw,relatime,user_id=0,group_id=0,allow_other,blksize=4096)
```