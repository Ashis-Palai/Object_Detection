1. OBJECTIVE - EXTRACTING DETAILS FROM A SET OF IMAGES :
********************************************************
********************************************************

Given a set of images where few details like (i.e ‘Device Name’, ‘REF’, ‘LOT’, ‘Qty’ and symbols.) are present inside the image, 
our project objective is to extract these details from the images and store in a Excel file.


2. PROJECT DESCRIPTION :
***********************
***********************
* What our application does: The application read given 7 images from the folrder and extract required details and store in a excel file.

* What technologies used and why : Combination of Python ,pytorch and Optical Character Recognition technologies are used in this project.
                                   As our objective is to extract symbols and texts, OCR technologies using Pytorch are quick and accurate for this type of project.

* Some of the challenges : Faced some challenges while performing Mosaic augmentation as the SSD(single shot detector) model did not predict as per the expectation.
                           Training data was very less to have higher accuracy, which was overcome by augmentaion techniques.


3. APPROCH TO SOLVE THE PROBLEM :
********************************
********************************

* Used Paddle-OCR to detect the texts inside given images.

* Generated synthetic data using images of the symbols by performing various techniques like Mosaic Augmentation, image resizing using Albumentations.

* Created 4000 synthetic data to train "SSD(Single Shot Detector)" and "Faster R-CNN models".

* Labeled  images using "LabelImg" tool (Pascal VOC format) so that we can get accurate positions of the bounding boxes.

* Customized "ssdlite320_mobilenet_v3_large" model to predict total of 10 classes of symbols including background class.

* With CUDA, dramatically sped up computing by harnessing the power of GPUs.

* Validated Mean Average Precision (mAP) Using the COCO Evaluator.

* Wrote logic using bounding box Coordinates to read the symbols from top to bottom so that we will get the symbols IDS in required order.

* Wrote logic to read the required texts like (i.e ‘Device Name’, ‘REF’, ‘LOT’, ‘Qty’ )  and save these details into an Excel file format.



4. HOW TO INSTALL AND RUN THE PROJECT :
***************************************
***************************************
* Unzip the provided ZIP file which contains (images folder, DKNSB_app.py ,logo_detection_ckpt_SSD_MODEL.pth , ReadMe.txt , requirements.txt )

* We need to install all the required packages & libraries mentioned in the requirements.txt files.

* Run the "DKNSB_app.py" file by making sure the path of "images" folder and path of "logo_detection_ckpt_SSD_MODEL.pth" are correct.

* An Excel file will be generated after running the project which will be stored as per the path.




