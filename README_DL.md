# 🐦 Scarecrow Bird Detection — Inference Script

This script runs inference on a test image directory using a trained YOLO model and saves both the visualized predictions (with bounding boxes) and a JSON file containing all predicted bounding boxes, confidence scores, and classes.

---

## 📂 Directory Structure


project/
├── inference.py
├── test_predictions.json
├── predicted_images/
└── runs/
    └── train/
        └── improved_model_v2_higher_resolution_3/
            └── weights/
                └── best.pt

The data is not saved in this ZIP file due to memory issues, this will have to be added manually for performing the inference.

---

## 🚀 How to Use

python
run_inference_and_save_predictions(
    model_path='runs/train/improved_model_v2_higher_resolution_3/weights/best.pt',
    image_dir=os.path.join(SCARECROW_PATH, 'test/images'),
    output_json='test_predictions.json'
)


This will:
- Load your YOLOv11 model from best.pt
- Run inference on all images in the specified test folder
- Save annotated images to predicted_images/
- Save detection results (coordinates, class, confidence) to test_predictions.json

---

## 📦 Dependencies

- Python ≥ 3.8
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- OpenCV
- Matplotlib (optional if you visualize)

Install requirements via:

bash
pip install ultralytics opencv-python


---

## 🛠 Function Details

python
def run_inference_and_save_predictions(
    model_path,             # Path to YOLO model weights (.pt)
    image_dir,              # Directory of input images
    output_dir="predicted_images",  # Where to save annotated images
    output_json="predictions.json", # Where to save the results
    conf_thresh=0.1         # Confidence threshold for predictions
)


---

## ✅ Output Example

predictions.json format:

json
[
  {
    "image": "example.jpg",
    "predictions": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.91,
        "class": 0
      },
      ...
    ]
  },
  ...
]


---

## ⚙️ Tested On

- *GPU:* NVIDIA A100
- *Batch size (during training):* 2
- *Inference memory usage:* ~2–4 GB (depending on image resolution and model size)
