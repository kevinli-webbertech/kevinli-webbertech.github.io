# Shell Scripting Ref

## variable

## function

## if

**if** and **exit**

exit code: 0 is success, other integer are errors. This is similar to c language.

```bash
if [ -z "$1" ]; then
  echo "No parameter";
  exit 22;
fi
```

**`$?`**

check exit code,

`echo $?`

## loop

**While** loop

```bash
x=1;
while [ $x -le 5 ]; do
  echo "Hello World"
  ((x=x+1))
done
```

## case


## Ref

- https://devhints.io/bash