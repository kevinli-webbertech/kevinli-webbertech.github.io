# Javascript Event Programming

## A toggle button example

In this example, we would like to make a toggle button. This button will not only show the content, but if we click on it again, it would hide the content.

In order to implement this example, we would use the following features of javascript we learned earlier,

* global variable
* function
* onclick event

To study this example, you can copy and paste the following html source code into a new html file you created. In my example, I created a file called "dom.html".

**dom.html**

```javascript
<html>
<head>
   <title>Using innerHTML property</title>
   <script>
      var displayed = false;

      function showIt() {
         document.getElementById("output").innerHTML = "Hello World";
         console.log("We flip this displayed flag on");
       
         displayed = true;
         console.log(displayed);
      }

      function toggle() {
         console.log("I am debugging in the toggle...");
         console.log(displayed);
         if (displayed) {
            // if it is true, then I want to hide it,
            document.getElementById("output").innerHTML="";
            displayed = false;
         } else {
            showIt()
         }
      }
   </script>
</head>

<body>
   <div id = "output" style="font-size: larger; font-family: 'Times New Roman', Times, serif; color: blueviolet;"> </div>

   <input type = "button" onclick = "toggle()" value = "display" />

</body>

<html>
```

In this example, we use a global variable as a flag to store a boolean value for us. The variable is called `displayed` and it was initially set to `false`. If we set the value then we flip it to `true`. Once we toggle the value back, we reset the value to empty string "" and we set the `displayed` to `false`.

**About Debugging**

Remember during the course that, if you type up the above code and run it and it does not work as expected, we can debug it by doing `console.log`.

How we do it is to right click on your html page, and click "Inspect", and then we go to "console" to check the printout messages from `console.log`.