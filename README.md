
# VisioAI: Image Captioning & Segmentation App

This project integrates **image captioning** and **image segmentation** using deep learning techniques. It enables users to upload an image and:

- Automatically **generate a descriptive caption**
- **Segment** and **highlight objects** (e.g., people, animals, objects) within the image

---

## 🔧 Features

- Caption generation using Custom Encoder and Decoder CNN model
- Object segmentation using **Mask R-CNN** (pretrained on COCO)
- Clean, interactive **Streamlit** web app UI
- Upload any image (person, object, animal, etc.) and get instant results

---

## Project Structure

```
ZIDIO_Task1/
│
├── app/                      # Main application directory
│   ├── app.py                # Streamlit UI logic
│   ├── utils.py              # Core functions (captioning & segmentation)
│   └── download_models.py    # Script to download model weights
│
├── requirements.txt          # All required Python libraries
├── README.md                 # Project overview and instructions
```

---

## Models Used

| Task         | Model                                                                                                      |
|--------------|------------------------------------------------------------------------------------------------------------|
| Captioning   | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)    |
| Segmentation | `Mask R-CNN (ResNet-50 FPN)` pretrained on the COCO dataset                                                |

---

## Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/Rishikesh4089/ZIDIO_Task1.git
cd ZIDIO_Task1/app
```

### 2. Install dependencies:
```bash
pip install -r ../requirements.txt
```

### 3. Download pre-trained models:
```bash
python download_models.py
```

### 4. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## Example Use Cases

| Input Image   | Caption                        | Segmentation              |
|---------------|--------------------------------|---------------------------|
| A dog and cat | “Cat and dog on a bench”       | Outlined animals          |
| Food          | “A plate of delicious pizza”   | Segmented food elements   |
| Person        | “A man working on a laptop”    | Person highlighted        |

---

## Datasets Used

- **Captioning**: MS COCO Dataset  
- **Segmentation**: Pascal VOC 2012

---

## 📄 Requirements

Make sure the following libraries are installed (handled via `requirements.txt`):
```
streamlit
torch
torchvision
transformers
Pillow
```

---

## 🔗 Repository

**GitHub**: [https://github.com/Rishikesh4089/VisioAI](https://github.com/Rishikesh4089/VisioAI)

---

## Project Status

- **Completed**
- Supports all general image types
- Tested locally and ready for internship submission
