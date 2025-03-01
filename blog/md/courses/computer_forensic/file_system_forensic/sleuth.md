# Sleuth Kit (TSK) Tutorial

The **Sleuth Kit (TSK)** is an open-source digital forensics toolkit used for analyzing disk images, file systems, and extracting digital evidence. It includes command-line tools for examining and recovering deleted files, listing file system metadata, and inspecting file structures.
It is a suite of commands/executables/tools provided.

## Key Takeaway

Sleuth Kit provides powerful tools for digital forensics, including:
- **File System Analysis** (`fsstat`, `fls`)
- **Metadata Extraction** (`istat`)
- **File Recovery** (`icat`, `ils`)
- **Timeline Creation** (`mactime`)


## Installation**

### **Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install sleuthkit
```

### **macOS (using Homebrew)**

```bash
brew install sleuthkit
```

### **Windows**
Download the binaries from the [official website](https://www.sleuthkit.org/) and install.

---

## Basic Commands**

Once installed, you can use TSK commands to analyze disk images and file systems.

### **List Partitions in a Disk Image**

```bash
mmls disk.img
```

- `disk.img` is a forensic disk image.
- This command outputs the partition table, showing partition start/end sectors.

### **Analyze a Specific Partition**

```bash
fsstat disk.img -o 2048
```

- `-o 2048` specifies the sector offset found using `mmls`.

### **List Files in a Partition**

```bash
fls -o 2048 disk.img
```

- Lists files and directories in the partition.

### **Extract a File**

```bash
icat -o 2048 disk.img 12345 > recovered_file.jpg
```

- Extracts file with inode `12345` and saves it as `recovered_file.jpg`.

### **Find Deleted Files**

```bash
ils -o 2048 disk.img
```

- Lists inodes of deleted files.

### **Recover Deleted Files**

```bash
icat -o 2048 disk.img 67890 > recovered.txt
```

- If `67890` is the inode of a deleted file, this command recovers it.

---

## **3. Case Study: Recovering Deleted Files from a Disk Image**
### **Step 1: Identify Partitions**

```bash
mmls disk.img
```

_Output Example:_

```
DOS Partition Table
Offset Sector: 0
Units are in 512-byte sectors

      Slot    Start        End          Length       Description
000:  0000000000   0000002047   0000002048   Primary Table (#1)
001:  0000002048   0001048575   0001046528   NTFS (0x07)
```

_The partition starts at sector `2048`._

### **Step 2: List Files**

```bash
fls -o 2048 disk.img
```

_Output Example:_

```
r/r 5-128-3: $AttrDef
r/r 6-144-3: $Bitmap
d/d 11-128-3: Documents
r/r 20-160-4: photo.jpg (deleted)
```

_The deleted file `photo.jpg` has inode `20`._

### **Step 3: Recover Deleted File**

```bash
icat -o 2048 disk.img 20 > recovered_photo.jpg
```

_Check the file to confirm recovery._

---

## **4. File System Analysis**
### **Check File System Type**
```bash
fsstat disk.img -o 2048
```
- Provides details about the file system type, size, and structures.

### **View Metadata of a Specific File**
```bash
istat disk.img -o 2048 20
```
- Shows metadata, including timestamps and size.

---

## **5. Searching for Specific Files**
### **Search for a Specific File Name**
```bash
ffind -o 2048 disk.img "confidential.txt"
```
- Returns inode of the file.

### **Find Files by Content**
```bash
blkls -o 2048 disk.img | strings | grep "password"
```
- Searches for the keyword "password" in unallocated space.

---

## **6. Timeline Analysis**
Create a forensic timeline to track file activities.

### **Step 1: Generate File System Metadata**
```bash
fls -r -o 2048 disk.img > filelist.txt
```
### **Step 2: Extract Timestamps**
```bash
mactime -b filelist.txt -d > timeline.csv
```
- The timeline contains **MAC times** (Modified, Accessed, Created).

---

