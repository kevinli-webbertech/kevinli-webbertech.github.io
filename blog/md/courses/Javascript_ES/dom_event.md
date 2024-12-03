# Javascript Event Programming

## A toggle button example

In this example, we would like to make a toggle button. This button will not only show the content, but if we click on it again, it would hide the content.

In order to implement this example, we would use the following features of javascript we learned earlier,

* global variable
* function
* onclick event

To study this example, you can copy and paste the following html source code into a new html file. In my example, I created a file called "dom.html".

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

## Use built-in event

Here's a simple HTML file to test the JavaScript `addNumbers` function:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Test Addition Function</title>
  <script>
    // JavaScript function to add two numbers
    function addNumbers(a, b) {
      return a + b;
    }

    // Function to handle form submission
    function handleAddition(event) {
      event.preventDefault(); // Prevent form reload
      const num1 = parseFloat(document.getElementById('number1').value);
      const num2 = parseFloat(document.getElementById('number2').value);
      const result = addNumbers(num1, num2);
      document.getElementById('result').textContent = `The sum is: ${result}`;
    }
  </script>
</head>
<body>
  <h1>Test Addition Function</h1>
  <form onsubmit="handleAddition(event)">
    <label for="number1">Enter first number:</label>
    <input type="number" id="number1" required><br><br>

    <label for="number2">Enter second number:</label>
    <input type="number" id="number2" required><br><br>

    <button type="submit">Add Numbers</button>
  </form>

  <h2 id="result"></h2>
</body>
</html>
```

### How to Use:
1. Copy and paste the code into a file, e.g., `test-addition.html`.
2. Open the file in a web browser.
3. Enter two numbers in the input fields and click the "Add Numbers" button.
4. The result will be displayed below the form. 

This is a simple way to test the `addNumbers` function interactively!