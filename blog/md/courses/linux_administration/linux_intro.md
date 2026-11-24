# Linux Introduction

## Who Created Linux?

<style>
  .linux-intro-portrait {
    float: right;
    width: 42%;
    max-width: 320px;
    margin: 0 0 1rem 1.5rem;
  }

  .linux-intro-portrait img {
    display: block;
    width: 100%;
    height: auto;
  }

  .linux-intro-portrait figcaption {
    margin-top: 0.5rem;
    font-size: 0.875em;
  }

  .linux-intro-timeline {
    clear: both;
  }

  @media (max-width: 600px) {
    .linux-intro-portrait {
      float: none;
      width: 100%;
      margin: 0 auto 1rem;
    }
  }
</style>

<figure class="linux-intro-portrait">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinuxCon_Europe_Linus_Torvalds_03_%28cropped%29.jpg" alt="Linus Torvalds speaking at LinuxCon Europe 2014 in Düsseldorf" width="320" loading="lazy">
  <figcaption><em>Linus Torvalds at LinuxCon Europe 2014. Photo by Krd; crop by Von Sprat, via <a href="https://commons.wikimedia.org/wiki/File:LinuxCon_Europe_Linus_Torvalds_03_(cropped).jpg">Wikimedia Commons</a>, licensed under <a href="https://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a>. Shown without further edits.</em></figcaption>
</figure>

**Linus Torvalds** is a software engineer born on December 28, 1969, in Helsinki, Finland. He became interested in programming as a child and studied at the University of Helsinki. His work on Linux helped establish one of the best-known collaborative open-source projects. Read more in the [Lemelson-MIT biography](https://lemelson.mit.edu/resources/linus-torvalds).

**Linus Torvalds created the Linux kernel in 1991** while studying at the University of Helsinki in Finland. He began it as a personal project for his Intel 386 computer. Other developers subsequently contributed, turning it into a collaborative open-source project.

Torvalds created the kernel, rather than every component of a complete Linux operating system. A usable system also includes libraries, command-line tools, services, and applications contributed by many projects, including GNU. The [Linux kernel introduction](https://www.kernel.org/doc/html/latest/admin-guide/README.html) describes its origins and capabilities.

<h2 class="linux-intro-timeline" id="when-did-linux-begin">When Did Linux Begin?</h2>

| Time | Event |
| --- | --- |
| 1991 | Torvalds began developing the Linux kernel while a student in Finland. |
| August 25, 1991 | He announced his operating-system project in the `comp.os.minix` Usenet group. This date is commonly celebrated as Linux's birthday. |
| September 17, 1991 | Linux version 0.01 was made available on the University of Helsinki FTP server. |

The project announcement and the first source release are separate milestones. See the [Linux anniversary page](https://www.linux.com/linux35th/) for the announcement date and the [USENIX history of Linux](https://www.usenix.org/system/files/login/issues/login_summer16_issue.pdf) for the first release timeline.

## What Is the Kernel?

The **kernel** is the central part of an operating system. It manages hardware resources and provides controlled interfaces that programs use to request services.

| Kernel Responsibility | What It Does | Example |
| --- | --- | --- |
| Process scheduling | Shares CPU time among running tasks. | A terminal, browser, and background service can make progress concurrently. |
| Memory management | Manages virtual memory and isolates process address spaces. | Programs receive their own virtual address spaces. |
| Device drivers | Communicates with hardware through drivers. | Programs can use disks, keyboards, and network adapters. |
| Filesystems | Provides access to files and directories through filesystem implementations. | A program reads a file stored on an ext4 filesystem. |
| Networking | Implements networking facilities and protocols. | A server exchanges data over TCP/IP. |
| Access control | Enforces permissions and other security checks. | A process cannot open a protected file without the required access. |

Linux uses a **monolithic kernel architecture with loadable modules**. Core services run in kernel space, while modules allow functionality such as drivers to be added without rebuilding the entire kernel. Ordinary applications run in user space and request kernel services through system calls.

For example, when an application reads a file, it typically uses a library function that invokes a system call. The kernel checks access, uses the filesystem and storage driver, and returns the requested data or an error.

## Kernel, Shell, and Distribution

| Term | Meaning | Examples |
| --- | --- | --- |
| Kernel | The core that manages hardware and system resources. | Linux |
| Shell | A command interpreter that starts programs and supports scripting. | Bash, Zsh |
| Distribution | A packaged operating system combining a kernel with tools, libraries, services, software repositories, and installation choices. | Debian, Ubuntu, Fedora, Kali Linux |
| Desktop environment | A collection of graphical desktop components and applications. | GNOME, KDE Plasma, Xfce |

Two distributions can use the Linux kernel while providing different package managers, default desktops, release schedules, and administration tools. A shell or desktop is not part of the kernel itself.

## Inspect Your Linux System

Show the kernel name:

```bash
uname -s
```

On Linux, this normally prints `Linux`.

Show the running kernel release:

```bash
uname -r
```

Identify the installed distribution on systems that provide `/etc/os-release`:

```bash
cat /etc/os-release
```

The kernel release and distribution version describe different parts of the system. Updating a distribution's kernel does not necessarily change its distribution version.

## Next Lesson

Continue with [Linux Distro](https://kevinli-webbertech.github.io/blog/html/courses/linux_administration/linux_distro.html) to compare Linux distributions.

## References

- [Linux kernel introduction and administration guide](https://www.kernel.org/doc/html/latest/admin-guide/README.html)
- [Linux anniversary and announcement date](https://www.linux.com/linux35th/)
- [USENIX: Linux at 25](https://www.usenix.org/system/files/login/issues/login_summer16_issue.pdf)
