import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import YolosForObjectDetection, YolosImageProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Loading the "YOLOS-tiny" model and its image processor
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Loading the COCO 2017 dataset with images with objects
# In this case we are loading the validation subset (5k images)
dataset = load_dataset("rafaelpadilla/coco2017", split="val")

# Preprocessing the image
def preprocess_image(image):
    inputs = processor(images=image, return_tensors="pt")
    return inputs

# This function is for displaying few images (range(10) for example)
# to visualise and test the code. In this case it will display 10 images
# with bounding boxes and labels (detected objects).
for i in range(10):  
    image = dataset[i]["image"]
    inputs = preprocess_image(image)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    plt.imshow(image)
    ax = plt.gca()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        if score > 0.5:
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="red", linewidth=2))
            ax.text(box[0], box[1], f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()

# This function will evaluate the model on the dataset we chose.
def evaluate_model(dataset, model, processor):
    all_boxes = []
    all_labels = []
    all_scores = []
    all_image_ids = []

    for i in tqdm(range(len(dataset))):
        try:
            image = dataset[i]["image"]
            
            # While testing, some images in dataset appeared to have
            # unsupported number of image dimensions. We need to check if
            # the image has 2 dimensions and if not, the problem 
            # image will be skipped and program will continue running.
            if len(image.size) != 2:
                raise ValueError("Unsupported number of image dimensions")
            
            inputs = preprocess_image(image)

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.5:
                    all_boxes.append(box.tolist())
                    all_labels.append(label.item())
                    all_scores.append(score.item())
                    all_image_ids.append(dataset[i]["image_id"])

        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    return all_boxes, all_labels, all_scores, all_image_ids

all_boxes, all_labels, all_scores, all_image_ids = evaluate_model(dataset, model, processor)

# Converting results to COCO format
coco_results = []
for box, label, score, image_id in zip(all_boxes, all_labels, all_scores, all_image_ids):
    coco_results.append({
        "image_id": image_id,
        "category_id": label,
        "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]],  # xywh format
        "score": score
    })

# Saving results to a JSON file
with open("yolos_tiny_results.json", "w") as f:
    json.dump(coco_results, f, indent=4)

# Loading ground truth (image annotations or in another words correct results for evaluation the model)
coco_gt = COCO('instances_val2017.json')

# Loading results
coco_dt = coco_gt.loadRes("yolos_tiny_results.json")

# Evaluating the model using COCO evaluation API from pycocotools
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Plotting Precision-Recall curve for visualisation of coco_eval results
precision = coco_eval.eval['precision']

iou_thresh = np.linspace(0.5, 0.95, np.size(precision, 0))

area_rng_lbl = ['all', 'small', 'medium', 'large']
area_rng = range(len(area_rng_lbl))

cat_ids = coco_gt.getCatIds()

plt.figure(figsize=(10, 8))

for idx, area in enumerate(area_rng):
    plt.plot(iou_thresh, np.mean(precision[:, :, :, area, cat_ids.index(1)], axis=(1, 2)), label=f'{area_rng_lbl[idx]}')
    
plt.xlabel('IoU Threshold')
plt.ylabel('Average Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()