\#  ZIDIO Internship Task: Image Captioning \& Segmentation App



This project combines image captioning and image segmentation using deep learning and computer vision. The application allows users to upload any image and:



\-  Automatically generate a caption

\-  Highlight segmented objects in the image







\##  Features



\-  Caption generation using BLIP (Salesforce model)

\-  Segmentation using Mask R-CNN

\-  Streamlit-based user interface

\-  Upload any type of image (person, object, animal, etc.)





---



\## Project Structure 



ZIDIO\_Task1/

│

├── app/                        # Main app 

│   ├── app.py                  # Streamlit UI 

│   ├── utils.py                # Functions for captioning and segmentation

│   └── download\_models.py      # Script to automatically download model weights

│

├── requirements.txt            # All Python libraries 

├── README.md  



\##  Models Used



| Task             | Model                                                                                                   |

|------------------|---------------------------------------------------------------------------------------------------------|

| Captioning       | \[`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) |

| Segmentation     | `Mask R-CNN (ResNet-50 FPN)` pretrained on COCO dataset                                                 |







\##  How to Run Locally



\###  Clone the repository:



```bash

git clone https://github.com/Rishikesh4089/ZIDIO\_Task1.git

cd ZIDIO\_Task1/app


Install dependencies


 pip install -r ../requirements.txt


Download the models:

 python download_models.py


Run the Streamlit app:


streamlit run app.py



| Input Image                         | Caption                         | Segmentation         |
| ------------------------------------| --------------------------------| ---------------------|
|  A dog and a cat sitting on a bench | “Cat and Dog on bench”         |  Objects outlined     |
|  Food Image                         | “A plate of delicious pizza”   |  Food segmented       |
|  Person Image                       | “A man working on a laptop”    |  Person highlighted   |



 Requirements
Install from requirements.txt:

text
Copy code
streamlit
torch
torchvision
transformers
Pillow


Datasets

Captioning: MS COCO  

Segmentation: Pascal VOC 2012


Repo: https://github.com/Rishikesh4089/ZIDIO_Task1




Status

 Completed and ready for internship submission.
 Supports captioning and segmentation for all general images.

---

 

