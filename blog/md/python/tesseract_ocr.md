# Tesseract Tutorial

OCR can help us do two things.

1/ Convert Image to Text

and what is more, 

2/ Convert PDF to Image.

Thus, it will ease our development and research time.
There are two approches, one is with pytesseract and second is with openCV.

## OCR Code

```
# Importing the os module to perform file operations
import os  
# Importing the pytesseract module to extract text from images
import pytesseract as tess  
# Importing the Image module from the PIL package to work with images
from PIL import Image  
# Importing the convert_from_path function from the pdf2image module to convert PDF files to images
from pdf2image import convert_from_path  

#This function takes a PDF file name as input and returns the name of the text file that contains the extracted text.
def read_pdf(file_name):   
    # Store all pages of one file here:
    pages = []

    try:
        # Convert the PDF file to a list of PIL images:
        images = convert_from_path(file_name)  

        # Extract text from each image:
        for i, image in enumerate(images):
          # Generating filename for each image
            filename = "page_" + str(i) + "_" + os.path.basename(file_name) + ".jpeg"  
            image.save(filename, "JPEG")  
          # Saving each image as JPEG
            text = tess.image_to_string(Image.open(filename))  # Extracting text from each image using pytesseract
            pages.append(text)  
          # Appending extracted text to pages list

    except Exception as e:
        print(str(e))

    # Write the extracted text to a file:
    output_file_name = os.path.splitext(file_name)[0] + ".txt"  # Generating output file name
    with open(output_file_name, "w") as f:
        f.write("\n".join(pages))  
      # Writing extracted text to output file

    return output_file_name

#print function returns the final converted text 
pdf_file = "sample.pdf"
print(read_pdf(pdf_file))
```

## Second example

```
# We import the necessary packages
#import the needed packages
import cv2
import os,argparse
import pytesseract
from PIL import Image
 
#We then Construct an Argument Parser
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",
                required=True,
                help="Path to the image folder")
ap.add_argument("-p","--pre_processor",
                default="thresh", 
                help="the preprocessor usage")
args=vars(ap.parse_args())
 
#We then read the image with text
images=cv2.imread(args["image"])
 
#convert to grayscale image
gray=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
 
#checking whether thresh or blur
if args["pre_processor"]=="thresh":
    cv2.threshold(gray, 0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
if args["pre_processor"]=="blur":
    cv2.medianBlur(gray, 3)
     
#memory usage with image i.e. adding image to memory
filename = "{}.jpg".format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
 
# show the output images
cv2.imshow("Image Input", images)
cv2.imshow("Output In Grayscale", gray)
cv2.waitKey(0)
```

## Conclusion

Which is better?

## REF

- https://github.com/tesseract-ocr/tesseract
- https://www.tutorialspoint.com/tesseract-ocr-with-java-with-examples
- https://www.geeksforgeeks.org/reading-text-from-the-image-using-tesseract/




**Conclusion:**

Both scripts serve different purposes, so which one is "better" depends on your specific needs. Here’s a comparison to help you decide which one to use:

**PDF to Text Conversion Script**
Purpose:

Converts PDF files to images and then extracts text from those images using Tesseract OCR.
Use Case:

Suitable for extracting text from PDF files that contain scanned images or non-selectable text.
Useful for multi-page PDFs where each page needs to be processed individually.
Pros:

Handles multi-page PDF documents efficiently.
Can deal with scanned documents where text is embedded as images.
Cons:

Requires handling multiple image files temporarily.
Might be slower due to the conversion of each PDF page to an image.

**Image Preprocessing and Text Extraction Script**
Purpose:

Reads an image file, preprocesses it (e.g., converts to grayscale, applies thresholding or blurring), and then extracts text using Tesseract OCR.
Use Case:

Suitable for extracting text from individual image files.
Useful for improving OCR accuracy on noisy or low-quality images through preprocessing.
Pros:

Allows for customizable preprocessing steps to improve OCR accuracy.
Faster for single image processing as there’s no need to convert multiple pages.
Cons:

Not suitable for multi-page PDFs directly (needs each page converted to an image first).