# services/module2/ocr_service.py
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

class OCRService:
    def __init__(self):
        model_name = "naver-clova-ix/donut-base-finetuned-docvqa"

        print("Loading Donut model... (first time may take ~1 min)")

        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.eval()
        self.device = "cpu"   # M2 works perfectly here
        self.model.to(self.device)

    def extract_text(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=128,
                num_beams=3,
                early_stopping=True
            )

        output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return output.strip()