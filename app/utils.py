from PIL import ImageDraw
import torch
import torchvision
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

# 🔹 Load BLIP model for captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 🔹 Load Mask R-CNN for segmentation
segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
segmentation_model.eval()

# 🔹 Transform image to tensor
segmentation_transform = transforms.Compose([transforms.ToTensor()])

# 🔹 Function to generate a real caption
def generate_caption(image):
    inputs = caption_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = caption_model.generate(**inputs)
    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# 🔹 Function to perform real segmentation
def segment_image(image):
    draw = ImageDraw.Draw(image)
    img_tensor = segmentation_transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = segmentation_model(img_tensor)[0]

    for i in range(len(prediction["boxes"])):
        score = prediction["scores"][i].item()
        if score > 0.7:  # confidence threshold
            box = prediction["boxes"][i]
            label = prediction["labels"][i].item()
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), f"Object {label}", fill="red")

    return image
