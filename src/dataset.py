import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class PoseDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # image_id -> file_name
        self.image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

        self.samples = []
        for ann in coco["annotations"]:
            if ann["num_keypoints"] == 0:
                continue

            image_id = ann["image_id"]
            if image_id not in self.image_id_to_file:
                continue

            img_path = os.path.join(image_dir, self.image_id_to_file[image_id])
            if not os.path.exists(img_path):
                continue

            keypoints = torch.tensor(ann["keypoints"], dtype=torch.float32)
            label = self._keypoints_to_label(keypoints)

            self.samples.append({
                "image_path": img_path,
                "keypoints": keypoints,
                "label": label
            })

        print(f"[INFO] Loaded {len(self.samples)} valid samples")

    def _keypoints_to_label(self, keypoints):
        kp = keypoints.view(-1,3)
        
        HEAD = 0
        LEFT_HAND = 9
        RIGHT_HAND = 10
        LEFT_HIP = 11
        RIGHT_HIP = 12

        hand_y = min(kp[LEFT_HAND][1], kp[RIGHT_HAND][1])
        hip_y = (kp[LEFT_HIP][1] + kp[RIGHT_HIP][1]) / 2
        head_y = kp[HEAD][1]

        # raise_hand: mano sopra testa
        if kp[LEFT_HAND][2] > 0 or kp[RIGHT_HAND][2] > 0:
            if hand_y < head_y:
                return 2  # raise_hand

        # sit vs stand: altezza dei fianchi rispetto alla testa
        torso_height = hip_y - head_y
        if torso_height < 50:  # soglia da aggiustare empiricamente
            return 1  # sit
        else:
            return 0  # stand
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        keypoints = sample["keypoints"].view(-1, 3)[:, :2].flatten()  # (x,y)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return keypoints, label
