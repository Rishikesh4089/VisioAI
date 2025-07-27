import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import os


MODEL_PATH = "../segmentation/segmentation.pkl" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def load_segmentation_model(model_path=MODEL_PATH, num_classes=21):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

_SEG_MODEL = load_segmentation_model()

def segment_image(image, min_score=0.5):
    draw = ImageDraw.Draw(image)
    seg_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = seg_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = _SEG_MODEL(img_tensor)[0]

    for i, box in enumerate(prediction["boxes"]):
        score = prediction["scores"][i].item()
        if score > min_score:
            x1, y1, x2, y2 = [float(x) for x in box]
            label_idx = prediction["labels"][i].item()
            label = VOC_CLASSES[label_idx] if label_idx < len(VOC_CLASSES) else str(label_idx)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(y1-12, 0)), f"{label} ({score:.2f})", fill="red")
    return image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../captioning/captioner.pt"  # Local saved weights
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
state_dict = torch.load(model_path, map_location=device)
caption_model.load_state_dict(state_dict, strict=False)
caption_model.eval()
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image):
    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = caption_model.generate(pixel_values, max_length=20)  # num_beams=1
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption