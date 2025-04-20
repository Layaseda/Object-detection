import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.image_processing_utils import BatchFeature
from torchmetrics.detection.mean_ap import MeanAveragePrecision


import wandb
wandb.init(project="ObjectDetection")


#Paths
image_dir = "images"  # extract images.zip 
train_json = "train.json"
val_json = "val.json"
test_json = "test.json"

# Directory to save trained model files
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

# Class Names 
category_names = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]



# Dataset class for loading COCO-style dataset
class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_path, processor):
        self.img_dir = img_dir
        self.processor = processor

        # Load the annotations from the COCO JSON file
        with open(ann_path, "r") as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # Create a mapping from image_id to its annotations
        self.ann_map = {}
        for ann in self.annotations:
            self.ann_map.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, img_info["file_name"])).convert("RGB")
        anns = self.ann_map.get(img_info["id"], [])

        # Prepare the target dictionary with annotations in COCO format
        target = {
            "image_id": img_info["id"],
            "annotations": [
                {
                    "bbox": ann["bbox"],
                    "category_id": ann["category_id"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": 0
                } for ann in anns
            ]
        }
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
            "targets_raw": target["annotations"]
        }


# Collate function to combine batches
def collate_fn(batch):
    return BatchFeature(data={
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
        "targets_raw": [x["targets_raw"] for x in batch],
    })


def move_to_device(batch, device):
    batch["pixel_values"] = batch["pixel_values"].to(device)
    batch["labels"] = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in label.items()}
        for label in batch["labels"]
    ]
    return batch



###########################################################################################


# Load Model and Processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50", num_labels=8, ignore_mismatched_sizes=True
)

# Freeze Backbone
for name, param in model.model.backbone.named_parameters():
    param.requires_grad = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)


#Load Data 
train_ds = CocoDetectionDataset(image_dir, train_json, processor)
val_ds = CocoDetectionDataset(image_dir, val_json, processor)
test_ds = CocoDetectionDataset(image_dir, test_json, processor)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn)



#Training Loop 
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = move_to_device(batch, device)
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
    print(f" Train Loss: {avg_train_loss:.4f}")

    

    #  Validation 
    model.eval()
    val_metric = MeanAveragePrecision(class_metrics=True)
    with torch.no_grad():
        for batch in val_loader:
            raw = batch["targets_raw"]
            batch = move_to_device(batch, device)
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            preds = processor.post_process_object_detection(outputs=outputs, target_sizes=[(1080, 1920)] * len(raw), threshold=0.5)
            targets = [
                {
                    "boxes": torch.tensor([[x, y, x + w, y + h] for x, y, w, h in [ann["bbox"] for ann in anns]]).to(device),
                    "labels": torch.tensor([ann["category_id"] for ann in anns]).to(device)
                } for anns in raw
            ]
            predictions = [
                {
                    "boxes": p["boxes"].to(device),
                    "scores": p["scores"].to(device),
                    "labels": p["labels"].to(device)
                } for p in preds
            ]
            val_metric.update(predictions, targets)

    val_result = val_metric.compute()
    wandb.log({
        "val_map": val_result["map"],
        "val_map_50": val_result["map_50"],
        "val_map_75": val_result["map_75"],
        "val_per_class_ap": wandb.Table(
            columns=["Class", "AP"],  
            data=[[category_names[i], float(x)] for i, x in enumerate(val_result["map_per_class"])]
        )
    })
    print(f" Val mAP: {val_result['map']:.4f} | mAP@50: {val_result['map_50']:.4f}")

    # Save Model
    torch.save(model.state_dict(), os.path.join(save_dir, f"detr_auair_epoch{epoch+1}.pth"))

    print(f" Model saved.")

#Save full final model
torch.save(model, os.path.join(save_dir, "detr_final_full_model.pth"))
print("Final full model saved.")



###########################################################################################


# Final Test Evaluation 
model.eval()
test_metric = MeanAveragePrecision(class_metrics=True)
with torch.no_grad():
    for batch in tqdm(test_loader, desc=" Test Evaluation"):
        raw = batch["targets_raw"]
        batch = move_to_device(batch, device)
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        preds = processor.post_process_object_detection(outputs=outputs, target_sizes=[(1080, 1920)] * len(raw), threshold=0.5)
        targets = [
            {
                "boxes": torch.tensor([[x, y, x + w, y + h] for x, y, w, h in [ann["bbox"] for ann in anns]]).to(device),
                "labels": torch.tensor([ann["category_id"] for ann in anns]).to(device)
            } for anns in raw
        ]
        predictions = [
            {
                "boxes": p["boxes"].to(device),
                "scores": p["scores"].to(device),
                "labels": p["labels"].to(device)
            } for p in preds
        ]
        test_metric.update(predictions, targets)

test_result = test_metric.compute()
wandb.log({
    "test_map": test_result["map"],
    "test_map_50": test_result["map_50"],
    "test_map_75": test_result["map_75"],
    "test_per_class_ap": wandb.Table(
        columns=["Class", "AP"], 
        data=[[category_names[i], float(x)] for i, x in enumerate(test_result["map_per_class"])]
    )
})
print(f"\n Test mAP: {test_result['map']:.4f} | mAP@50: {test_result['map_50']:.4f} | mAP@75: {test_result['map_75']:.4f}") 