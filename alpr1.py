# Importing libraries

import numpy as np #Numerical computing library for array manipulation.
import cv2   #OpenCV library for image and video processing.
import tensorflow as tf # Core library for building and training machine learning models.
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array  #Submodule of Keras for image preprocessing tasks like loading and converting images.
import pytesseract as pt #Library for optical character recognition (OCR) using Tesseract engine.
import easyocr #Python library for OCR with pre-trained models for various languages.
from time import time 
import google.generativeai as genai # to get vehicle type and license plate category using gemini api
from dotenv import load_dotenv #Library for loading environment variables from a .env file.
from io import BytesIO  #Library for working with input/output operations, used here with BytesIO for in-memory data handling.
import tempfile  #Library for creating temporary files.
import os #Library for interacting with the operating system (e.g., creating directories).
from PIL import Image #Python Imaging Library for image manipulation
import warnings
warnings.filterwarnings("ignore")

# loading pretrained saved model for detecting the license plate
model= tf.keras.models.load_model(r'D:\DL_Project\ANPR\models\object_detection.h5')


# license plate detection
def object_detection(path,filename):
#    Performs object detection on an image to identify and locate an object of interest.
#    Args:
#        path (str): The path to the image file.
#        filename (str): The filename of the image.
#    Returns:
#        numpy.ndarray: The coordinates of the detected object (xmin, xmax, ymin, ymax).

    # read image
    image=load_img(path)
    # Convert the image to a NumPy array with 8-bit unsigned integers (0-255)
    image=np.array(image,dtype=np.uint8)
    # Load a resized version of the image for the model (assuming model expects 224x224)
    image1=load_img(path,target_size=(224,224))
    
    # Data preprocessing:
    # - Convert the resized image to a NumPy array.
    # - Normalize pixel values to the range 0-1.
    image_arr_224=img_to_array(image1)/255.0
    # Get the original image's dimensions
    h,w,d = image.shape
    # Reshape the array for model input (assuming batch size of 1)
    test_arr=image_arr_224.reshape(1,224,224,3)
    # Make predictions using the pre trained and saved model
    coords= model.predict(test_arr)
    
    # Denormalize the model's output coordinates to match the original image size
    denorm= np.array([w,w,h,h])
    coords=coords*denorm
    coords =coords.astype(np.int32) # Convert to integers for drawing
    # Draw a bounding box around the detected object
    xmin,xmax,ymin,ymax = coords[0]
    pt1=(xmin,ymin)
    pt2=(xmax,ymax)
    cv2.rectangle(image,pt1,pt2,(0,255,0),2) # Green rectangle

    # Convert the image from RGB (Pillow format) to BGR (OpenCV format)
    image_bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # Save the image with the bounding box
    cv2.imwrite('./predict_images/{}'.format(filename),image_bgr)
    # print(coords)
    return coords

# extract number from image

def OCR(path,filename):
    # Performs Optical Character Recognition (OCR) on an image to extract the license plate number.
    # Args:
    #     path (str): The path to the image file containing the vehicle.
    #     filename (str): The filename of the image.
    # Returns:
    #     str: The extracted license plate number, or an empty string if unable to read the plate.
    
    # Initialize the EasyOCR reader for English text extraction
    reader = easyocr.Reader(['en'])
    # Load the image and convert it to a NumPy array
    img=np.array(load_img(path))
    # Detect the coordinates of the license plate in the image by calling the function object_detection
    coords=object_detection(path,filename)
    # Extract the bounding box coordinates of the license plate
    xmin,xmax,ymin,ymax=coords[0]
    # Crop the image based on the license plate coordinates
    license_plate = img[ymin:ymax,xmin:xmax]
    # Generate a unique filename with timestamp for saving the cropped image
    timestamp = int(time())
    # Append the timestamp to the filename
    filename_with_timestamp = "{}_{}.jpg".format(filename, timestamp)
    # Save the cropped image for potential debugging or analysis
    cv2.imwrite('./license_plates/{}'.format(filename_with_timestamp),license_plate)
    # Convert the cropped image to grayscale
    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to enhance contrast and isolate characters
    _,license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
    # Perform OCR using EasyOCR
    output = reader.readtext(license_plate_thresh)
    print(output)
    # Initialize an empty string to store the extracted text
    text=''
    # Iterate through the OCR output  
    for res in output:
    # If there is only one OCR result, assign its text to 'text'
     if(len(output))==1:
          text=res[1]
    # If there are multiple OCR results and the text length is greater than 6 and confidence score is above 0.2, assign the text to 'text'
     if len(output) >1 and len(res[1]) >6 and res[2] >0.2:
           text=res[1]
    # text=pt.image_to_string(license_plate)
    # print("License Plate no: ",text)
    return text


def getRTO(number):
    # Retrieves the Regional Transport Office (RTO) name for a given vehicle registration number in Kerala, India.

    # Args:
    #     number (str): The vehicle registration number.

    # Returns:
    #     str: The RTO name corresponding to the vehicle registration number, or an empty string if not found.
    
    # Extract the first four letters from the vehicle registration number, excluding any spaces.
    first_4letter=number.replace(" ", "")[:4]
    # Create a dictionary mapping RTO codes to their corresponding names.
    rto_list={ "KL01": "RTO TRIVANDRUM",
        "KL02": "RTO KOLLAM",
        "KL03": "RTO PATHANAMTHITTA",
        "KL04": "RTO ALAPPUZHA",
        "KL05": "RTO KOTTAYAM",
        "KL06": "RTO IDUKKI",
        "KL07": "RTO ERNAKULAM",
        "KL08": "RTO THRISSUR",
        "KL09": "RTO PALAKKAD",
        "KL10": "RTO MALAPPURAM",
        "KL11": "RTO KOZHIKKODE",
        "KL12": "RTO WAYANAD",
        "KL13": "RTO KANNUR",
        "KL14": "RTO KASARAGOD",
        "KL15": "RTO NATIONALISED SECTOR",
        "KL16": "RTO ATTINGAL",
        "KL17": "RTO MUVATTUPUZHA",
        "KL18": "RTO VADAKKARA",
        "KL19": "SRTO PARASSALA",
        "KL20":	"SRTO NEYYATTINKARA",
        "KL21":	"SRTO NEDUMANGADU",
        "KL22":	"SRTO KAZHAKUTTOM",
        "KL23":	"SRTO KARUNAGAPPALLY",
        "KL24":	"SRTO KOTTARAKKARA",
        "KL25":	"SRTO PUNALUR",
        "KL26":	"SRTO ADOOR",
        "KL27":	"SRTO THIRUVALLA",
        "KL28":	"SRTO MALLAPPALLY",
        "KL29":	"SRTO KAYAMKULAM",
        "KL30": "SRTO CHENGANNUR",
        "KL31": "SRTO MAVELIKKARA",
        "KL32": "SRTO CHERTHALA",
        "KL33": "SRTO CHANGANASSERY",
        "KL34":	"SRTO KANJIRAPPALLY",
        "KL35": "SRTO PALA",
        "KL36": "SRTO VAIKKOM",
        "KL37": "SRTO VANDIPERIYAR",
        "KL38": "SRTO THODUPUZHA",
        "KL39":	"SRTO TRIPUNITHURA",
        "KL40": "SRTO PERUMBAVOOR",
        "KL41": "SRTO ALUVA",
        "KL42": "SRTO NORTH PAROOR",
        "KL43": "SRTO MATTANCHERRY",
        "KL44":	"SRTO KOTHAMANGALAM",
        "KL45": "SRTO IRINJALAKKUDA",
        "KL46": "SRTO GURUVAYOOR",
        "KL47": "SRTO KODUNGALLUR",
        "KL48": "SRTO VADAKKANCHERRY",
        "KL49":	"SRTO ALATHURA",
        "KL50": "SRTO MANNARKKAD",
        "KL51": "SRTO OTTAPPALAM",
        "KL52": "SRTO PATTAMBI",
        "KL53": "SRTO PERINTHALMANNA",
        "KL54":	"SRTO PONNANI",
        "KL55": "SRTO TIRUR",
        "KL56": "SRTO KOYILANDY",
        "KL57": "SRTO KODUVALLY",
        "KL58": "SRTO THALASSERY",
        "KL59":	"SRTO THALIPARAMBA",
        "KL60": "SRTO KANHANGAD",
        "KL61": "SRTO KUNNATHUR",
        "KL62": "SRTO RANNI",
        "KL63": "SRTO ANGAMALY",
        "KL64":	"SRTO CHALAKKUDY",
        "KL65": "SRTO TIRUR",
        "KL66": "SRTO KUTTANADU",
        "KL67": "SRTO UZHAVOOR",
        "KL68": "SRTO DEVIKULAM",
        "KL69":	"SRTO UDUMBANCHOLA",
        "KL70": "SRTO CHITTUR",
        "KL71": "SRTO NILAMBUR",
        "KL72": "SRTO MANANTHAVADY",
        "KL73": "SRTO SULTHANBATHERY",
        "KL74":	"SRTO KATTAKKADA",
        "KL75": "SRTO THRIPRAYAR",
        "KL76": "SRTO NANMANDA",
        "KL77": "SRTO PERAMBRA",
        "KL78": "SRTO IRITTY",
        "KL79":	"SRTO VELLARIKUNDU",
        "KL80": "SRTO PATHANAPURAM",
        "KL81": "SRTO VARKALA",
        "KL82": "SRTO CHADAYAMANGALAM",
        "KL83": "SRTO KONNI",
        "KL84":	"SRTO KONDOTTY",
        "KL85": "SRTO RAMANATTUKARA",
        "KL86": "SRTO PAYYANNUR",
    }
    # Initialize an empty string to store the RTO name
    rto_name=''
    # Check if the first four letters are not empty
    if first_4letter!='':
        # Use the `get` method of the dictionary to retrieve the RTO name.
        # If the key (first four letters) is not found, it returns None.
        rto_name=rto_list.get(first_4letter)
        
    # Return the RTO name, or an empty string if not found
    return rto_name


# to load all the environment variables
load_dotenv()
# retrieves the value of an environment variable named GOOGLE_API_KEY and
# identifies you as a user of the GenAI API and grants you access to its capabilities.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini Pro Vision API And get response
def get_gemini_repsonse(input,image,prompt):
    
    # Fetches a response from the Gemini language model based on the provided input, image, and prompt.
    # Args:
    #     input (str): The initial text input for the model.
    #     image (list): A list containing the image data (assumed to be pre-processed).
    #     prompt (str): The prompt or instructions for the model.
    # Returns:
    #     str: The textual response generated by the Gemini model.
    
    # Load the Gemini GenerativeModel
    model=genai.GenerativeModel('gemini-pro-vision')
    
    # Generate content using the model
    response=model.generate_content([input,image[0],prompt])
    
    # Extract the text from the response
    return response.text


def readImage(image_path):
    
    #     """
    # Reads and preprocesses an image from a specified file path.

    # Args:
    #     image_path (str): The path to the image file.

    # Returns:
    #     list: A list containing a dictionary with image details:
    #         - "mime_type": The MIME type of the image (e.g., "image/jpeg").
    #         - "data": The binary image data.
    # """
    
    # Open the image file
    img = Image.open(image_path)
    # Convert image to RGB mode if it has an alpha channel
    # (Alpha channel represents transparency)
    if img.mode == 'RGBA':
        # Convert to RGB for better compatibility
        img = img.convert('RGB')
    # Convert AVIF to JPEG (or PNG) if needed
    if img.format != "JPEG":
        # Create a temporary file to save the JPEG image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as temp_filename:
            img.save(temp_filename.name, format="JPEG")  # Save as temporary JPEG
            with open(temp_filename.name, 'rb') as f:
                image_bytes = f.read() # Read binary data from temporary file
    else:
        # If already JPEG, just read the bytes
        with BytesIO() as output:
            # Save to in-memory buffer
            img.save(output, format='JPEG')
            # Get the bytes from the BytesIO buffer
            image_bytes = output.getvalue()
    # Store the image bytes in a list with MIME type information
    img_parts = [
            {
                "mime_type": 'image/jpeg',  # MIME type for JPEG images
                "data": image_bytes # Binary image data
            }
        ]
    return img_parts
    
input_prompt=""" Provide the details of the vehicle in the picture in below format Vehicle type: type of the vehicle
License Plate No: License Plate Number of the vehicle
Type of the License Plate  : identify the category of vehicle based on the color type of the license plates in india
"""


def getVehicleDetails(image_path):
#        """
#    Extracts vehicle details from an image using Gemini.
#    Args:
#        image_path (str): The path to the image file.
#    Returns:
#        str: The textual response from Gemini containing vehicle details.
#    """
    # Preprocess the image. Load and prepare image for Gemini
    image_data=readImage(image_path)
    # Fetch response from Gemini. Generate response with image
    response=get_gemini_repsonse(input_prompt,image_data,"")
    # Calls the get_gemini_response function to fetch a response from Gemini, providing:
    # An input_prompt (assumed to be defined elsewhere).
    # The preprocessed image_data.
    # An empty string ("") for the prompt, indicating no additional instructions for Gemini. The response is stored in the response variable.
    
    return response


#extract the license plate number from gemini response
def extractResponse(response):
    # Split the response by newlines
    lines = response.splitlines()

    # Find the line containing "vehicle number:"
    for line in lines:
        if "License Plate No:" in line:
            # Split the line by colon
            parts = line.split(": ")
            # Extract the license plate number
            license_plate_number = parts[1]
            break  # Exit the loop after finding the first occurrence
        
    # return the license plate number
    return license_plate_number

image_path=r'./images/car1.jpg'
filename=r'car1.jpg'
number=OCR(image_path,filename)
res=getVehicleDetails(image_path)
number=extractResponse(res)
rto=getRTO(number)
print("License plate Number extracted using OCR: ",number)
print(res.strip())
print(rto)

