import json
import os
import torch
import pandas as pd
from monai.transforms import (EnsureChannelFirstd, Compose, Lambdad, NormalizeIntensityd, RandCoarseShuffled, RandRotated, RandZoomd,
                              Resized, ToTensord, LoadImaged)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class QaTa(Dataset):
    """
    Custom PyTorch Dataset for handling image and text data.
    """

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224, 224], image_dir='Images', gt_dir='Ground-truths'):
        super(QaTa, self).__init__()
        print(csv_path)
        self.mode = mode

        if not csv_path or not os.path.exists(csv_path):
            raise ValueError(f"Invalid csv_path: {csv_path}")

        self.data = pd.read_csv(csv_path)
        if 'Image' not in self.data.columns or 'Description' not in self.data.columns:
            raise ValueError("CSV file must contain 'Image' and 'Description' columns.")

        # Extract only the file names from the 'Image' column
        self.image_list = [os.path.basename(img) for img in self.data['Image']]
        self.image_list = [img.replace("Images\\", "") for img in self.image_list]
        #self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        if mode == 'train':
            self.image_list = self.image_list[:int(0.8 * len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.8 * len(self.caption_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8 * len(self.image_list)):]
            self.caption_list = self.caption_list[int(0.8 * len(self.caption_list)):]

        self.root_path = root_path
        self.image_size = image_size
        self.image_dir = image_dir  # Initialize the image directory
        self.gt_dir = gt_dir  # Initialize the ground truth directory

        if not tokenizer:
            raise ValueError("A valid tokenizer must be provided.")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path, self.image_dir, self.image_list[idx].replace('mask_', ''))
        #image = os.path.normpath(self.image_list[idx])  # normalizes slashes
        #gt = os.path.join(self.root_path, self.gt_dir, self.image_list[idx])
        gt = os.path.join(self.root_path, self.gt_dir, f"mask_{self.image_list[idx]}")
        caption = self.caption_list[idx]
        print(self.root_path, self.image_dir, self.image_list[idx].replace('mask_', ''))
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        if not os.path.exists(gt):
            raise FileNotFoundError(f"Ground truth file not found: {gt}")

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                max_length=24, 
                                                truncation=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image': image, 'gt': gt, 'token': token, 'mask': mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt == 255, 1, 0)
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)} 

        return ([image, text], gt)

    def transform(self, image_size=[224, 224]):
        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ])
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ])
        return trans