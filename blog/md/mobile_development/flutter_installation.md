# Mobile Development - Flutter Framework

# What is Flutter?

**Takeaway**

* Multi-platform mobile app framework
* Written in Dart

Flutter is an open-source UI software development kit created by Google. It can be used to develop cross platform applications from a single codebase for the web, Fuchsia, Android, iOS, Linux, macOS, and Windows. First described in 2015 Flutter was released in May 2017. Flutter is used internally by Google in apps such as Google Pay and Google Earth as well as by other software developers including ByteDance and Alibaba.

Flutter consists of both a UI language and a rendering engine. When a Flutter application is compiled, it ships with both the UI code and the rendering engine, which is about 4 MB compressed. This is in contrast to many other UI frameworks that rely on a separate rendering engine and only ship the UI code, such as native Android apps which rely on the device-level Android SDK or HTML/JavaScript Web apps that rely on the user's HTML engine and JavaScript engine. Flutter's complete control of its rendering pipeline makes supporting multiple platforms simpler as it only needs the platform to support running native code such as via the Android Java Native Interface rather than support Flutter's UI model in its entirety.

On May 6, 2020, the Dart software development kit (SDK) version 2.8 and Flutter 1.17.0 were released, adding support for the Metal API as well as new Material widgets and network tracking development tools.

On September 8, 2021, Dart 2.14 and Flutter 2.5 were released with the latest version of Material Design called Material You.

On May 12, 2022, Flutter 3 and Dart 2.17 were released with support for all desktop platforms as stable.

## Key concepts involved

* Widgets, stateless and stateful
A widget describes the logic, interaction, and design of a UI element with an implementation similar to React. Unlike other cross-platform toolkits such as React Native and Xamarin which draw widgets using native platform components, Flutter renders widgets itself on a per-pixel basis.

* Dart Language

Flutter apps are written in the Dart language. Release versions of Flutter apps on all platforms use ahead-of-time (AOT) compilation except for on the Web where code is transpiled to JavaScript[27] or WebAssembly.

* Flutter Engine

Flutter supports two different rendering engine engine back-ends: Google's Skia graphics library and Flutter's own engine called Impeller. Impeller is enabled by default on iOS and is currently in beta on Android. The engine interfaces with platform-specific SDKs such as those provided by Android and iOS to implement features like accessibility, file and network I/O, native plugin support, etc.[32]

* Foundation library

The Foundation library, written in Dart, provides basic classes and functions that are used to construct applications using Flutter, such as APIs to communicate with the engine.

## Installation

- Step 1 Install linux tools and commands

To develop Flutter on Linux:

Verify that you have the following tools installed: `bash`, `file`, `mkdir`, `rm`, `which`

```
$ which bash file mkdir rm which
/bin/bash
/usr/bin/file
/bin/mkdir
/bin/rm
which: shell built-in command

```

Install the following packages: `curl`, `git`, `unzip`, `xz-utils`, `zip`, `libglu1-mesa`

```
$ sudo apt-get update -y && sudo apt-get upgrade -y;
$ sudo apt-get install -y curl git unzip xz-utils zip libglu1-mesa
```

- Step 2 Install Android Studio (IDE)

To develop Android apps:

Install the following prerequisite packages for Android Studio: 

libc6:i386, libncurses5:i386, libstdc++6:i386, lib32z1, libbz2-1.0:i386

```
$ sudo apt-get install \
    libc6:i386 libncurses5:i386 \
    libstdc++6:i386 lib32z1 \
    libbz2-1.0:i386
```

Install Android Studio 2023.2.1 (Iguana) or later to debug and compile Java or Kotlin code for Android. Flutter requires the full version of Android Studio.

- Step 3 IDE - Visual studio Code

* Visual Studio Code 1.77 or later with the Flutter extension for VS Code.
* IntelliJ IDEA 2023.2 or later with the Flutter plugin for IntelliJ.

Note: The Flutter team recommends installing Visual Studio Code 1.77 or later and the Flutter extension for VS Code. This combination simplifies installing the Flutter SDK.


- Step 3 Install the Flutter SDK

Linux or VSCode (skipped here)

For our development work, please come and load this VM image so that you can start to use it.

[download flutter vm](url)

- Step 4 Configure Android development

Configure the Android toolchain in Android Studio

To create Android apps with Flutter, verify that the following Android components have been installed.

* Android SDK Platform, API 34.0.5
* Android SDK Command-line Tools
* Android SDK Build-Tools
* Android SDK Platform-Tools
* Android Emulator

If you haven't installed these, or you don't know, continue with the following procedure.

Otherwise, you can skip to the next section.

First time using Android Studio
Current Android Studio User
Launch Android Studio.

The Welcome to Android Studio dialog displays.

Follow the Android Studio Setup Wizard.

Install the following components:

* Android SDK Platform, API 34.0.5
* Android SDK Command-line Tools
* Android SDK Build-Tools
* Android SDK Platform-Tools
* Android Emulator

- Step 5 Configure your target Android device - Set up the Android emulator

To configure your Flutter app to run in an Android emulator, follow these steps to create and select an emulator.

1. Enable VM acceleration on your development computer.

2. Start Android Studio.

3. Go to the Settings dialog to view the SDK Manager.

* If you have a project open, go to Tools > Device Manager.

* If the Welcome to Android Studio dialog displays, click the More Options icon that follows the Open button and click Device Manager from the dropdown menu.

4. Click Virtual.

5. Click Create Device.

 The Virtual Device Configuration dialog displays.

6. Select either Phone or Tablet under Category.

7. Select a device definition. You can browse or search for the device.

8. Click Next.

9. Click x86 Images.

10. Click one system image for the Android version you want to emulate.

* If the desired image has a Download icon to the right of the Release Name, click it.

The SDK Quickfix Installation dialog displays with a completion meter.

* When the download completes, click Finish.

11. Click Next.

 The Virtual Device Configuration displays its Verify Configuration step.

12. To rename the Android Virtual Device (AVD), change the value in the AVD Name box.

13. Click Show Advanced Settings and scroll to Emulated Performance.

14. From the Graphics dropdown menu, select Hardware - GLES 2.0.

This enables hardware acceleration and improves rendering performance.

15. Verify your AVD configuration. If it is correct, click Finish.

To learn more about AVDs, check out Managing AVDs.

16. In the Device Manager dialog, click the Run icon to the right of your desired AVD. The emulator starts up and displays the default canvas for your selected Android OS version and device.

- Step 6 Check your development setup - Run Flutter doctor

The flutter doctor command validates that all components of a complete Flutter development environment for Linux.

Open a shell.

To verify your installation of all the components, run the following command.

`$ flutter doctor`

As you chose to develop for Android, you do not need all components. If you followed this guide, the result of your command should resemble:

```
Running flutter doctor...
Doctor summary (to see all details, run flutter doctor -v):
[✓] Flutter (Channel stable, 3.22.0, on Ubuntu 20.04 (LTS), locale en)
[✓] Android toolchain - develop for Android devices (Android SDK version 34.0.5)
[!] Chrome - develop for the web
[✓] Android Studio (version 2023.3)
[!] Linux toolchain - develop for Linux desktop
[✓] VS Code (version 1.89)
[✓] Connected device (1 available)
[✓] Network resources

! Doctor found issues in 2 categories.
```

**ref** 

https://docs.flutter.dev/get-started/install
https://docs.flutter.dev/get-started/install/linux/android

## Ref
https://developers.google.com/learn/topics/flutter
https://codelabs.developers.google.com/codelabs/flutter-codelab-first#0
https://www.tutorialspoint.com/flutter/index.htm
https://en.wikipedia.org/wiki/Flutter_(software)
https://www.geeksforgeeks.org/flutter-tutorial/

