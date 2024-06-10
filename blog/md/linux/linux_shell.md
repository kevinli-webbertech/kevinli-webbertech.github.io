# Shell Scripting Ref

## variable

```
name="John"
echo $name  # see below
echo "$name"
echo "${name}!"
```

Generally quote your variables unless they contain wildcards to expand or command fragments.

```
wildcard="*.txt"
options="iv"
cp -$options $wildcard /tmp
```

## String Quotes

```
String quotes
name="John"
echo "Hi $name"  #=> Hi John
echo 'Hi $name'  #=> Hi $name
```

## Shell execution

```
echo "I'm in $(pwd)"
echo "I'm in `pwd`"  # obsolescent
```

## Strict mode

```
set -euo pipefail
IFS=$'\n\t'
```

# Conditional Execution

```
git commit && git push
git commit || echo "Commit failed"
```

## Brace Expansion

```
echo {A,B}.js
{A,B} Same as A B
{A,B}.js Same as A.js B.js
{1..5} Same as 1 2 3 4 5
{{1..3},{7..9}} Same as 1 2 3 7 8 9
```

*example*

```
xiaofengli@xiaofenglx:~$ echo {1..5}
1 2 3 4 5
xiaofengli@xiaofenglx:~$ echo {{1..5},{6..9}}
1 2 3 4 5 6 7 8 9
```

## Conditionls

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

```
if [[ -z "$string" ]]; then
  echo "String is empty"
elif [[ -n "$string" ]]; then
  echo "String is not empty"
fi
```

## Loop

**for in loop**

```
for i in /etc/rc.*; do
  echo "$i"
done
```

**for in with range**

```
for i in {1..5}; do
    echo "Welcome $i"
done
```

**for in with step**

```
for i in {5..50..5}; do
    echo "Welcome $i"
done
```

**while loop**

```bash
x=1;
while [ $x -le 5 ]; do
  echo "Hello World"
  ((x=x+1))
done
```

**C-like for loop**

```bash
for ((i = 0 ; i < 100 ; i++)); do
  echo "$i"
done
```

**Read lines from file**

```bash
while read -r line; do
  echo "$line"
done <file.txt
```

## Function

```bash
get_name() {
  echo "John"
}

echo "You are $(get_name)"
```

another example,

```bash
myfunc() {
    echo "hello $1"
}
# Same as above (alternate syntax)
function myfunc {
    echo "hello $1"
}
myfunc "John"
```

**Returning values**

```bash
myfunc() {
    local myresult='some value'
    echo "$myresult"
}
result=$(myfunc)
```

**Raising errors**

```bash
myfunc() {
  return 1
}
if myfunc; then
  echo "success"
else
  echo "failure"
fi
```

**Arguments**

`$#` Number of arguments
`$*` All positional arguments (as a single word)
`$@` All positional arguments (as separate strings)
`$1` First argument
`$`_ Last argument of the previous command

## Ref

- <https://devhints.io/bash>
