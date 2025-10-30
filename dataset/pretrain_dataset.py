import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import pre_caption


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, imgcap_root, max_words=100):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.imgcap_root = imgcap_root

    def __len__(self):
        return len(self.ann)

    def get_image_path(self, image_name):
        if 'NWPU' in image_name:
            return os.path.join(self.imgcap_root, image_name)
        elif 'RSICD' in image_name:
            return os.path.join(self.imgcap_root, image_name)
        elif 'Det10' in image_name:
            return os.path.join(self.imgcap_root, image_name)
        else:
            return os.path.join(self.image_root, image_name)

    def __getitem__(self, index):
        ann = self.ann[index]

        # 处理 caption
        if isinstance(ann['caption'], list):
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        # 处理图像路径与图像
        image_name = ann['image']
        image_path = self.get_image_path(image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # 处理区域框和区域描述（如果有）
        region_boxes = ann.get("region_boxes", [])  # 允许为空
        region_texts = [pre_caption(text, self.max_words) for text in ann.get("region_texts", [])]
        # 转成 Tensor 类型
        import torch
        region_boxes = torch.tensor(region_boxes, dtype=torch.float32)
        region_boxes = torch.clamp(region_boxes, 0.0, 1.0)
        assert len(region_boxes) == len(region_texts), f"Inconsistent region data at index {index}"

        return image, caption, region_boxes, region_texts

