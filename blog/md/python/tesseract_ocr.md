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




**Tesseract OCR Tutorial: Comparing Two Approaches for Text Extraction**

Optical Character Recognition (OCR) is a technology that can convert different types of documents, such as scanned paper documents, PDF files, or images captured by a digital camera, into editable and searchable data. In this tutorial, we will explore two approaches to OCR using Tesseract: one with Pytesseract and another with OpenCV. Each method has its strengths and weaknesses, and the best choice depends on your specific needs.

**Introduction to Tesseract OCR**
Tesseract is an open-source OCR engine that can recognize and convert printed text into digital text. It's widely used for various applications, from digitizing printed documents to enabling text-based searches in images.

**Approach 1: Using Pytesseract to Convert PDFs to Text**
The first approach involves converting PDF files to images and then extracting text from those images using Pytesseract. This method is particularly useful for handling multi-page PDFs where each page needs to be processed individually.

**Code Example**

import os
import pytesseract as tess
from PIL import Image
from pdf2image import convert_from_path

def read_pdf(file_name):
    pages = []
    try:
        images = convert_from_path(file_name)
        for i, image in enumerate(images):
            filename = "page_" + str(i) + "_" + os.path.basename(file_name) + ".jpeg"
            image.save(filename, "JPEG")
            text = tess.image_to_string(Image.open(filename))
            pages.append(text)
    except Exception as e:
        print(str(e))
    output_file_name = os.path.splitext(file_name)[0] + ".txt"
    with open(output_file_name, "w") as f:
        f.write("\n".join(pages))
    return output_file_name

pdf_file = "sample.pdf"
print(read_pdf(pdf_file))

**Pros and Cons**
Pros:
Handles Multi-Page PDFs: Efficiently processes multi-page PDF documents.
Suitable for Scanned Documents: Can extract text from scanned PDFs where text is embedded as images.
Cons:
Temporary Image Files: Requires handling multiple temporary image files.
Slower Processing: Conversion of each PDF page to an image can be time-consuming.

**Approach 2: Using OpenCV for Image Preprocessing and Text Extraction**
The second approach involves reading an image file, preprocessing it (e.g., converting to grayscale, applying thresholding or blurring), and then extracting text using Pytesseract. This method is ideal for individual image files where preprocessing can significantly improve OCR accuracy.

**Code Example**

import cv2
import os, argparse
import pytesseract
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image folder")
ap.add_argument("-p", "--pre_processor", default="thresh", help="the preprocessor usage")
args = vars(ap.parse_args())

images = cv2.imread(args["image"])
gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

if args["pre_processor"] == "thresh":
    cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
if args["pre_processor"] == "blur":
    cv2.medianBlur(gray, 3)

filename = "{}.jpg".format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

cv2.imshow("Image Input", images)
cv2.imshow("Output In Grayscale", gray)
cv2.waitKey(0)

**Pros and Cons**
Pros:
Customizable Preprocessing: Allows for customizable preprocessing steps to improve OCR accuracy.
Faster for Single Images: Processes individual images faster as there is no need to convert multiple pages.
Cons:
Not Suitable for Multi-Page PDFs: Needs each page converted to an image first, not efficient for multi-page documents.
Conclusion: Which Approach is Better?

**Choosing the best approach depends on your specific requirements:**

Use Pytesseract for PDFs: If you need to handle multi-page PDF documents, especially scanned ones, the Pytesseract approach is more suitable. It efficiently processes each page and extracts text.

Use OpenCV for Images: If you're dealing with individual images and need to improve OCR accuracy through preprocessing, the OpenCV approach is better. It allows for customizable preprocessing steps like thresholding and blurring.

Both methods leverage Tesseract's powerful OCR capabilities, but their use cases differ. By understanding the strengths and limitations of each approach, you can select the best one for your project.

