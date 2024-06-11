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