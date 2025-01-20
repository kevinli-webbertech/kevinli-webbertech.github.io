# **Password Cracking Example Using Kali Linux**

Password cracking is the process of trying to recover or "crack" passwords from stored data (such as password hashes) by using a variety of methods, such as **brute-force attacks**, **dictionary attacks**, and **rainbow table attacks**.

In this example, I'll walk you through **cracking hashed passwords** using two popular tools in Kali Linux: **John the Ripper** and **Hashcat**.

---

### **1. Cracking Password Hashes with John the Ripper**

**John the Ripper (JTR)** is a popular and versatile password-cracking tool. It supports various hashing algorithms like **MD5**, **SHA-1**, **SHA-256**, **LM**, and others. Here's an example of how to use **John the Ripper** to crack password hashes.

#### **Step 1: Install John the Ripper (if needed)**
- John the Ripper comes pre-installed in **Kali Linux**, but if you need to install it manually, use the following command:
  ```bash
  sudo apt-get install john
  ```

#### **Step 2: Prepare Password Hashes**

Let’s say you have a file containing **MD5 password hashes**. The file (e.g., `passwords.txt`) contains hashes like this:
```
5f4dcc3b5aa765d61d8327deb882cf99
```

This is the MD5 hash for the password `"password"`.

#### **Step 3: Crack the Hash with John the Ripper**

Run **John the Ripper** to crack the hash:

```bash
john --format=raw-md5 passwords.txt
```

Explanation:
- `--format=raw-md5`: This specifies that the hashes in the file are in **raw MD5 format**. John the Ripper will attempt to crack the MD5 hashes.
- `passwords.txt`: The file containing the MD5 hashes.

**John the Ripper** will attempt different passwords from its internal dictionary and using brute force until it finds the correct one.

#### **Step 4: Check the Cracked Password**

Once John the Ripper finishes, you can view the cracked password by running:

```bash
john --show passwords.txt
```

The output will show the cracked password(s) and the associated hash(es):

```
passwords.txt
5f4dcc3b5aa765d61d8327deb882cf99:password
```

---

### **2. Cracking WPA/WPA2 Passwords with Hashcat**

**Hashcat** is another powerful tool for cracking hashed passwords. It is optimized for **GPU acceleration**, which makes it much faster than CPU-based tools for many types of attacks.

In this example, we'll use **Hashcat** to crack a **WPA2 Wi-Fi password**.

#### **Step 1: Install Hashcat (if needed)**
Hashcat is also pre-installed in **Kali Linux**, but if you need to install it manually, use the following command:
```bash
sudo apt-get install hashcat
```

#### **Step 2: Capture WPA2 Handshake**

Before cracking WPA2 passwords, you need to capture the **handshake** between a device and the Wi-Fi router. This can be done using **aircrack-ng** or **airodump-ng**.

1. Start **airodump-ng** to monitor your wireless network:
   ```bash
   sudo airodump-ng wlan0mon
   ```

2. Once you see a target network, run **airodump-ng** again to capture the handshake when a client connects to the network:
   ```bash
   sudo airodump-ng --bssid <target-AP-MAC> -c <channel> --write handshake wlan0mon
   ```

   This will capture the handshake and save it as `handshake.cap`.

#### **Step 3: Crack WPA2 Handshake with Hashcat**

Once you have the `.cap` file containing the handshake, use **Hashcat** to crack the password.

```bash
hashcat -m 2500 -a 0 handshake.cap /path/to/wordlist.txt
```

Explanation:
- `-m 2500`: Specifies the hash mode for WPA2 (this is the mode used for cracking WPA/WPA2 handshakes).
- `-a 0`: This sets the attack mode to **straight attack** (using a wordlist).
- `handshake.cap`: The file containing the captured WPA2 handshake.
- `/path/to/wordlist.txt`: The wordlist file (dictionary) that Hashcat will use to attempt to crack the password.

**Hashcat** will go through the wordlist, attempting each password until it either cracks the password or exhausts the list.

#### **Step 4: Check Cracked Password**

If Hashcat successfully cracks the password, it will display it in the terminal. You can also use this command to see the cracked password:

```bash
hashcat -m 2500 --show handshake.cap
```

Output example:
```
<hash>  password123
```

---

### **3. Cracking Hashes with Brute Force**

While **dictionary attacks** are the most efficient, there may be cases where the password is not in the wordlist. In such cases, you can use a **brute-force attack**, where Hashcat will try every possible combination of characters until it finds the correct one.

For example, to perform a brute-force attack with **Hashcat**, use the following command:

```bash
hashcat -m 0 -a 3 hashes.txt ?a?a?a?a?a?a?a?a
```

Explanation:
- `-m 0`: Specifies the hash mode (MD5 in this case).
- `-a 3`: Selects the **brute-force attack** mode.
- `?a?a?a?a?a?a?a?a`: Specifies a mask where `?a` means any **alphanumeric character** (lowercase, uppercase, digits, and special characters), and `?a?a?a?a?a?a?a?a` defines an 8-character password.

The brute-force attack will try all combinations of 8 characters, but this can take a long time depending on the length and complexity of the password.

---

### **4. Cracking Windows Password Hashes**

If you have access to a Windows system's **SAM** (Security Account Manager) file, which contains the password hashes, you can use **John the Ripper** or **Hashcat** to crack them.

#### **Step 1: Extract the Hashes**

You can extract the password hashes from a Windows system using tools like **Mimikatz** or **Samdump2**.

For example, with **Samdump2**, you can extract the hashes from the SAM file like this:

```bash
samdump2 SYSTEM SAM > hashes.txt
```

#### **Step 2: Crack the Hashes**

Once you have the password hashes, use **John the Ripper** or **Hashcat** to crack them.

With **John the Ripper**:
```bash
john hashes.txt
```

With **Hashcat**:
```bash
hashcat -m 1000 -a 0 hashes.txt /path/to/wordlist.txt
```

Explanation:
- `-m 1000`: Specifies the hash mode for Windows NTLM hashes.
- `-a 0`: Straight mode (using a wordlist attack).

---

### **Conclusion**

Password cracking is an essential part of **penetration testing** and **security auditing**, and Kali Linux provides powerful tools like **John the Ripper** and **Hashcat** to perform these tasks. Whether you're cracking **MD5 hashes**, **WPA2 passwords**, or **Windows NTLM hashes**, Kali Linux offers a versatile platform for testing password strength and improving security.