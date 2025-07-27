
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
| Captioning   | `Encoder-Decoder with CNN + RNN (LSTM)`                                                                    |
| Segmentation | `Mask R-CNN (ResNet-50 FPN)` pretrained on the COCO dataset                                                |

---

## 🧠 Model Architectures

This section outlines the architecture of the deep learning models used for image captioning and image segmentation in the project.

---

### 🖼️ Image Captioning Model

#### 📌 **Architecture Used**  
**Encoder–Decoder architecture with CNN-RNN**

- **Encoder:** Pretrained **ResNet-50**
  - Extracts image features.
  - Final fully connected layer removed.
  - Output passed through a linear layer and batch normalization to produce fixed-size embeddings.
  
- **Decoder:** LSTM-based language model
  - Takes image feature vector as input.
  - Generates captions word by word using an embedding layer, LSTM, and linear classifier.

#### ⚙️ **Key Components**

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| ResNet-50        | CNN backbone for feature extraction (pretrained, frozen)                    |
| Linear Layer     | Projects features to embedding space                                        |
| BatchNorm1d      | Normalizes the embedding vector                                             |
| LSTM             | Sequential decoder that generates a caption                                |
| Word Embedding   | Converts input tokens into vectors                                          |
| Linear Classifier| Outputs vocabulary distribution per time step                              |

---

### 🖍️ Image Segmentation Model

#### 📌 **Architecture Used**  
**Mask R-CNN with ResNet-50 + FPN**

- **Base Model:** Mask R-CNN (Facebook AI Research)
- **Backbone:** ResNet-50 with Feature Pyramid Network (FPN)
- **Function:** Performs object detection and instance-level segmentation.

#### ⚙️ **Key Components**

| Component          | Description                                                                   |
|--------------------|-------------------------------------------------------------------------------|
| ResNet-50          | Feature extractor for input images                                            |
| Feature Pyramid Net| Enhances feature maps at multiple scales                                      |
| RPN                | Region Proposal Network that suggests candidate object regions                |
| RoIAlign           | Precisely aligns features with proposed bounding boxes (better than RoIPool) |
| Classifier         | Predicts object class                                                         |
| BBox Regressor     | Refines bounding boxes                                                        |
| Mask Head          | Generates pixel-wise segmentation mask for each detected object               |



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
