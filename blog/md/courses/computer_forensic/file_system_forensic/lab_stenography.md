# Lab - Stenography 

### ref

- https://www.garykessler.net/library/fsc_stego.html

- https://forensicstats.org/steganography/

- https://forensicstats.org/steganography/

## Lab

### Step 1: Understanding Pixel Data

Every pixel in an image consists of three color values: Red (R), Green (G), and Blue (B). Each of these values is stored as an integer from 0 to 255.

    • Example pixel: (R: 173, G: 201, B: 124)

The least significant bit (LSB) of these values determines tiny changes that are invisible to the human eye. By tweaking these LSBs, we can hide binary data inside an image.

### Step 2: Preparing Your Files

    1. Choose a PNG image
        ◦ Use any small, simple image (less than 500x500 pixels is ideal).
        ◦ Open it in a basic image editor like Microsoft Paint (Windows) or Preview (Mac).
    2. Prepare Your Secret Message
        ◦ Convert your message to binary using an online ASCII to binary converter or a calculator.
        ◦ Example:
            ▪ "NYC" → 01001110 01011001 01000011
        ◦ Add a stop sequence (11111111 11111110) to mark the end.

### Step 3: Hiding the Message

    1. Open the PNG file in an Image Editor
        ◦ Zoom in and select individual pixels for editing.
        ◦ Check the RGB values of a few pixels.
    2. Modify the Least Significant Bit (LSB)
        ◦ Convert the pixel values to binary.
        ◦ Example: A pixel with values (173, 201, 124) in binary:
            ▪ 173 → 10101101
            ▪ 201 → 11001001
            ▪ 124 → 01111100
        ◦ Modify the last bit of each color channel to match the binary message.
        ◦ Example (hiding "NYC" in two pixels):
            ▪ Before: (173, 201, 124) → 10101101 11001001 01111100
            ▪ After: (172, 200, 125) → 10101100 11001000 01111101
    3. Save the Image
        ◦ Export the modified image as a new PNG file.

### Step 4: Extracting the Hidden Message

    1. Open the PNG File Again
        ◦ Pick the same pixels you modified.
        ◦ Convert their RGB values back to binary.
    2. Extract the Last Bits
        ◦ Read the LSB from each value and reconstruct the message.
        ◦ Stop when you reach the stop sequence (11111111 11111110).
    3. Convert Back to Text
        ◦ Group the extracted bits into 8-bit segments.
        ◦ Convert them back to characters using an ASCII table.
