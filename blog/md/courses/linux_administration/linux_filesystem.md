# **Linux File Systems Overview**

## Takeaway

* Linux top directoreis
* Linux file system
* Automounting and /etc/fstab

## Top directories

It looks like you have a Python script with Huffman encoding, Red-Black tree implementation, and LVM/RAID automation. Here’s how you can add **Linux top-level directories and their uses** into your script:

### **Linux Top-Level Directories and Their Uses**

| **Directory** | **Description** |
|--------------|----------------|
| `/` | Root directory, base of the file system. |
| `/bin` | Essential binary executables (e.g., `ls`, `cp`). |
| `/sbin` | System binaries (e.g., `fdisk`, `reboot`), typically for root. |
| `/etc` | Configuration files for the system and applications. |
| `/home` | Home directories for users (e.g., `/home/user`). |
| `/var` | Variable data (logs, databases, mail, etc.). |
| `/usr` | User-related programs, libraries (`/usr/bin`, `/usr/lib`). |
| `/opt` | Optional software installed manually. |
| `/tmp` | Temporary files (cleared on reboot). |
| `/dev` | Device files (e.g., `/dev/sda1` for a disk). |
| `/mnt` | Mount point for external storage. |
| `/media` | Automounted removable media (USB, CD-ROM). |
| `/proc` | Virtual file system for process and kernel information. |
| `/sys` | System information, kernel parameters. |
| `/boot` | Bootloader files (e.g., `vmlinuz`, `initrd`). |
| `/root` | Home directory of the root user. |
| `/lib` | Shared system libraries required for binaries in `/bin` and `/sbin`. |

### **Exploring `/proc` in Linux**

The `/proc` directory is a **virtual file system** that provides real-time system information. It doesn’t store actual files but contains dynamic system data.

---

### **Common `/proc` Files and Their Uses**

| **File/Directory** | **Description** |
|------------------|----------------|
| `/proc/cpuinfo` | Information about the CPU (model, cores, speed). |
| `/proc/meminfo` | Memory usage details (total, free, buffers, swap). |
| `/proc/uptime` | System uptime in seconds. |
| `/proc/loadavg` | System load averages. |
| `/proc/version` | Kernel version and compiler details. |
| `/proc/filesystems` | Supported file systems in the kernel. |
| `/proc/mounts` | Mounted file systems. |
| `/proc/[PID]` | Directory for each running process with details (e.g., `/proc/1` for init). |
| `/proc/net` | Network-related information. |

>Hint: The above are all files. As a matter of fact. `File` is a key in learning linux system.

#### **Automation with Python**

```python
def read_proc_file(file):
    """Reads and displays content from a /proc file."""
    try:
        with open(file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File {file} not found."

def system_info():
    """Displays key system information from /proc."""
    print("\nCPU Info:")
    print(read_proc_file("/proc/cpuinfo").split("\n")[0])  # Show only first line

    print("\nMemory Info:")
    print(read_proc_file("/proc/meminfo").split("\n")[0:5])  # Show first 5 lines

    print("\nUptime:")
    print(read_proc_file("/proc/uptime"))

    print("\nKernel Version:")
    print(read_proc_file("/proc/version"))

# Call system_info() to print details
if __name__ == "__main__":
    system_info()
```

## File Systems

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

- **List available file systems**

  ```bash
  cat /proc/filesystems
  ```

- **Check file system type of a partition**

  ```bash
  lsblk -f
  ```

- **Format a partition with ext4**

  ```bash
  sudo mkfs.ext4 /dev/sdX
  ```

- **Mount a file system**

  ```bash
  sudo mount /dev/sdX /mnt
  ```

- **Check disk usage**

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

- **Example 1**

```shell
xiaofengli@xiaofenglx:~$ mount |grep ntfs
/dev/sdb1 on /mnt/ntfs type fuseblk (rw,relatime,user_id=0,group_id=0,allow_other,blksize=4096)
```

