# LD (gcc linker)

`ld` is the **GNU linker** in Linux, responsible for linking object files (`.o`) and libraries to produce an executable or another object file. It is usually called by the compiler (e.g., `gcc` or `clang`) rather than being used directly.

## **What `ld` Does**

1. **Combines multiple object files** (`.o`) into a single executable.
2. **Resolves symbol references** across files.
3. **Links against libraries** (static `.a` or shared `.so`).
4. **Assigns addresses** to symbols and sections.
5. **Produces an output binary** (executable or another object file).

## **Basic Usage**

```sh
ld -o output_file input_files.o
```

However, `ld` is a **low-level** tool. Typically, you'd use `gcc` or `clang`, which internally call `ld`:

```sh
gcc -o my_program file1.o file2.o -lmylib
```

This automatically invokes `ld` with the correct system libraries.

## **Common `ld` Options**
- `-o <file>`: Specifies output file name.
- `-r`: Generates a relocatable object file instead of an executable.
- `-L<dir>`: Adds a directory to the library search path.
- `-l<lib>`: Links with a library (e.g., `-lm` for `libm.so`).
- `-T <script>`: Uses a custom linker script.

## **Example: Manually Linking**

Compile object files:

```sh
gcc -c file1.c file2.c
```

Manually link:

```sh
ld -o my_program file1.o file2.o -lc
```

(`-lc` links with the C standard library.)

