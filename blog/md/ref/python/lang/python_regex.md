# Python Ref - Regex

## Regex

### regex match

`re.match(pattern, string, flags=0)`

### regex search

`asdf`

### regex place

`re.sub(pattern, repl, string, max=0)`

### optional flags

`flags`: You can specify different flags using bitwise OR (|). These are modifiers, which are listed in the table below.
The modifiers are specified as an optional flag. You can provide multiple modifiers using exclusive OR (|), as shown previously and may be represented by one of these âˆ’

`re.I`: Performs case-insensitive matching.

`re.L`: Interprets words according to the current locale. This interpretation affects the alphabetic group (\w and \W), as well as word boundary behavior(\b and \B).

`re.M`: Makes $ match the end of a line (not just the end of the string) and makes ^ match the start of any line (not just the start of the string).

`re.S`: Makes a period (dot) match any character, including a newline.

`re.U`: Interprets letters according to the Unicode character set. This flag affects the behavior of \w, \W, \b, \B.

`re.X`: Permits "cuter" regular expression syntax. It ignores whitespace (except inside a set [] or when escaped by a backslash) and treats unescaped # as a comment marker.

### Control characters

* Basics

Control characters, ( + ? . * ^ $ ( ) [ ] { } | \ ), to escape control characters, put a backslash \ in the front.

`^:` begining of the line

`$:` end of the line

`.:` any single char except newline, using m option to make it match new line as well.

`[]`: match single characters in the branch, which defines a ring.

`[^]`: match single characters not in the brackets

`re*`: 0 or more occurence of expr `re`

`re+`: 1 or more occurence of expr `re`

`re?`: 0 or 1 occurence of expr `re`

`re{n}`: match exact n repeating of re

`re{n,}`: match n and more repeating

`re{n,m}`: match between n and m times

`a|b`: matches either a or b

`(re)`: groups regular expressions and remembers matched text.

`\w`: word

`\W`: nonword

`\s`: whitespace, such as \t\n\r\f

`\S`: not whitespace

`\d`: digit

`\D`: non digit

`\A`: Match begining of string

`\Z`: Match end of string. If new line exists, it matches just before newline.

`\z`: Match end of string.

`\G`: match point where last match finished

`\B`: non word boundary

`\n,\t`: match new line, carriage returns, tabs...etc

`\1...\9`: matches nth grouped subexpression.

`\10`: Matches nth grouped subexpression if it matched already. Otherwise refers to the octal representation of a character code.

* Advanced

(?imx): Temporarily toggles on i, m, or x options within a regular expression. If in parentheses, only that area is affected.

(?-imx): Temporarily toggles off i, m, or x options within a regular expression. If in parentheses, only that area is affected.

(?: re): Groups regular expressions without remembering matched text.

(?imx: re): Temporarily toggles on i, m, or x options within parentheses.

(?-imx: re): Temporarily toggles off i, m, or x options within parentheses.

(?#...): Comment.

(?= re): Specifies position using a pattern. Doesn't have a range.

(?! re): Specifies position using pattern negation. Doesn't have a range.

(?> re): Matches independent pattern without backtracking.

Example:

```python
#!/usr/bin/python
import re

## Greedy repetion example

line = "<python>perl>"

matchObj = re.match( r'<.*>', line, re.M|re.I)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
  
else:
   print "No match!!"

## Non greedy repetion example, get the first one, and stop

matchObj = re.match( r'<.*?>', line, re.M|re.I)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
  
else:
   print "No match!!"
   
line = "<<python>perl>"

matchObj = re.match( r'<.*?>', line, re.M|re.I)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```

* Grouping

```python
(\D\d)+
([Pp]ython(, )?)+
```

* Backreferences

```python
([Pp])ython&\1ails

Match python&pails or Python&Pails
	
(['"])[^\1]*\1

Single or double-quoted string. \1 matches whatever the 1st group matched. \2 matches whatever the 2nd group matched, etc.
```

* specify options for subexpr

?i means the same as the re.I option is for global, but ?i can be applied to subexpr or string before it.

```python
## apply options works the same as the $
print("testing ?i");
line = "ruby"
matchObj = re.match(r'R(?i)uby', line, re.M)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```

?! to negate some symbol after it.

```python
print("testing ?!");
line = "Python"
matchObj = re.match(r'Python(?!!)', line, re.M)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```

?= means followed by something,

```python
print("testing ?=");
line = "Python!"
matchObj = re.match(r'Python(?=!)', line, re.M)

if matchObj:
   print "matchObj.group() : ", matchObj.group()
else:
   print "No match!!"
```