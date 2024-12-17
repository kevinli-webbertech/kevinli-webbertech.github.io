# Control Flow

Javascript control flow is similar to other programming language such as Java, Python and C. The core idea are the following parts,

* if/else, switch branches
* loop, break, continue
* switch and goto labels

## if/else

First, let us take a look at `if/else` structure.

A program can have the following situations,

* only use `if(...){}`
* use `if(...){} else{}` structure
* use `if(...){} else if(...){}` structure
* use `if(...){} else if(...){} and else{}` structure

**Example 1**

```javascript
<!DOCTYPE html>
<html>
<body>

<h2>JavaScript if(){}_else{}/else_if(){}/else{}</h2>

<p>This program is to test our if else logics</p>

<p id="demo">Good Evening!</p>

<script>
    var currentTime = new Date().getHours(); // based in 24 hrs format
    document.write("<h2>This is an complete example of using, only if</h2>");
    if (currentTime < 13) { // fasting at 4 PM
        document.getElementById("demo").innerHTML = "You see this message";
    } else if (currentTime < 18) {
       
    }
    else if (currentTime < 23 ) {
         // let say currentTime is 22.
        // skip the above blocks,
        // it is actually  currentTime>=18 && currentTime< 23

        document.write("debug here....");
        console.log("debugging on the web page");
    }

</script>

</body>
</html>
```

**Example 2**

```javascript
<!DOCTYPE html>
<html>
<body>

<h2>JavaScript if(){}_else{}/else_if(){}/else{}</h2>

<p>This program is to test our if else logics</p>

<p id="demo">Good Evening!</p>


<script>
    var currentTime = new Date().getHours(); // based in 24 hrs format
    document.write("<h2>This is an complete example of using, only if</h2>");
    if (currentTime < 16) { // fasting at 4 PM
        document.getElementById("demo").innerHTML = "Reminder to eat your brunch!!";
    } else {
        document.getElementById("demo").innerHTML = "Reminder to eat your dinner!!";
    }
</script>

</body>
</html>
```

**Example 3**

```javascript
<!DOCTYPE html>
<html>
<body>

<h2>JavaScript if(){}_else{}/else_if(){}/else{}</h2>

<p>This program is to test our if else logics</p>

<p id="demo">Good Evening!</p>


<script>
    var currentTime = new Date().getHours(); // based in 24 hrs format
    document.write(<h2>This is an complete example of using, if else + else if + else</h2>);
    if (currentTime < 12) {
        document.getElementById("demo").innerHTML = "Good morning!";
    } else if (currentTime < 18) { // sunset time
        document.getElementById("demo").innerHTML = "Good afternoon!";
    } else {
        document.getElementById("demo").innerHTML = "Good evening!";
    }

  
</script>

<script>
  document.write("<h2>This is an complete example of using, only if</h2>");
    if (currentTime < 16) { // fasting at 4 PM
        document.getElementById("demo").innerHTML = "Reminder to eat your brunch!!";
    } else {
        document.getElementById("demo").innerHTML = "Reminder to eat your dinner!!";
    }
    </script>
</body>
</html>
```

