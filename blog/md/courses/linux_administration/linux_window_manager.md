# Linux Display Managers, Window Managers, and Desktop Sessions

**Many Linux display managers support both X11 and Wayland, but not all of them.** Compatibility depends on the display manager, its configuration, and the desktop sessions installed by your distribution.

A **display manager** (such as GDM, SDDM, or LightDM) handles the graphical login screen and starts desktop sessions. It is different from a **window manager** (such as i3, Openbox, or KWin), which manages application windows. A desktop environment, such as GNOME, KDE Plasma, or Xfce, provides a broader collection of desktop components and applications.

## 1. Linux Display Manager Compatibility

The X11 and Wayland columns describe the types of desktop sessions the manager can launch, rather than the protocol used by its own login screen.

| Display Manager | X11 Sessions | Wayland Sessions | Notes |
| --- | --- | --- | --- |
| **GDM** (GNOME) | Version and distribution dependent | Yes | Strong integration with GNOME Wayland. Newer GNOME releases have reduced or removed native GNOME X11 session support; check the installed GDM and desktop packages. |
| **SDDM** (commonly used with KDE Plasma) | Yes | Yes | Supports both session types. Its Wayland greeter is documented as experimental. |
| **LightDM** | Yes | Limited / configuration dependent | Primarily uses X11 greeters; launching a Wayland session depends on the installed session and configuration. |
| **LXDM** | Yes | No native support | Designed for X11 sessions. |
| **XDM** | Yes | No native support | Traditional X11 display manager. |
| **greetd** | Yes, with a suitable session command | Yes, with a suitable session command | A flexible login daemon rather than a display server; the selected greeter and session command determine the graphical environment. |

**Key point:** A display manager can run its login screen using X11 while launching a Wayland desktop session. The login screen and the user's desktop do not necessarily use the same display protocol.

## 2. How X11 and Wayland Sessions Work

![A display manager launches either an X11 session with Xorg and a window manager, or a Wayland session with Mutter, KWin, or Sway.](https://kevinli-webbertech.github.io/blog/images/linux_administration/display-manager-sessions.svg)

For example, SDDM can present an X11 login screen and then start a KDE Plasma Wayland session. Alternatively, SDDM can use a Wayland compositor for its greeter. These are separate configuration choices; SDDM's documentation labels the Wayland greeter experimental. See the [SDDM configuration manual](https://github.com/sddm/sddm/blob/develop/data/man/sddm.conf.rst.in).

Under X11, a window manager manages windows through the X server. Under Wayland, the compositor acts as the display server and manages windows and composition.

**Xwayland** lets compatible X11 applications run within a Wayland desktop. This does not make the desktop an X11 session or turn an X11 window manager into a native Wayland compositor.

## 3. Check Your Linux System

Run these commands in a terminal inside your graphical desktop session.

Find your current session protocol:

```bash
echo "$XDG_SESSION_TYPE"
```

The output is usually `x11` or `wayland`. A text console may report `tty`, and the variable can be unset in other contexts.

On a system using systemd, find your active display manager:

```bash
systemctl status display-manager
```

This reports the service's status when the distribution provides a `display-manager` service alias. A system started through a console login may have no display manager.

See which session definitions are installed:

```bash
ls /usr/share/xsessions/
ls /usr/share/wayland-sessions/
```

These are common locations for X11 and Wayland session definitions. A directory may be absent if that session type is not installed. greetd can instead start a configured session command directly.

## 4. Window Manager and Compositor Compatibility

Window managers and compositors have different compatibility rules from display managers.

| Window Manager / Compositor | Native X11 Session | Native Wayland Session | Notes |
| --- | --- | --- | --- |
| **i3** | Yes | No | An X11 tiling window manager. |
| **Sway** | No | Yes | A Wayland compositor with an i3-style workflow. |
| **Openbox** | Yes | No | An X11 window manager. |
| **KWin** (KDE) | Version and package dependent | Yes | X11 support is provided by the separate `kwin_x11` project in releases that still support it; current KWin development targets Wayland. |
| **Mutter** (GNOME) | Older releases only | Yes | Older versions supported native X11 sessions. Current Mutter is a Wayland compositor and can run X11 applications through Xwayland. |
| **Hyprland** | No | Yes | A Wayland compositor; X11 applications can use Xwayland. |

Traditional X11 window managers generally cannot run as native Wayland compositors. A Wayland desktop requires a compositor that implements the Wayland protocol.

GNOME 49 disabled its dedicated X11 session by default. Do not assume that an X11 session is available on newer systems simply because an older compatibility table lists it. See the [GNOME 49 developer release notes](https://release.gnome.org/49/developers/) and the [current Mutter documentation](https://mutter.gnome.org/).

For KDE, the [KWin project split announcement](https://mail.kde.org/pipermail/distributions/2025-March/001567.html) explains the separate X11 and Wayland implementations. Session availability depends on your Plasma release and distribution packages.

## 5. Choosing a Display Manager

For GNOME, use GDM for its desktop integration. For KDE Plasma, SDDM is a common choice. For standalone Wayland environments such as Sway or Hyprland, consider greetd with a suitable greeter and session command.

When troubleshooting, first identify whether the problem affects the **login screen**, the **desktop session**, or an **individual application**. Then check the display manager, window manager or compositor, and session protocol separately.

## References

- [SDDM project](https://github.com/sddm/sddm)
- [SDDM configuration manual](https://github.com/sddm/sddm/blob/develop/data/man/sddm.conf.rst.in)
- [greetd project](https://github.com/kennylevinsen/greetd)
- [GNOME 49 developer release notes](https://release.gnome.org/49/developers/)
- [Mutter documentation](https://mutter.gnome.org/)
- [KWin X11 and Wayland split announcement](https://mail.kde.org/pipermail/distributions/2025-March/001567.html)
