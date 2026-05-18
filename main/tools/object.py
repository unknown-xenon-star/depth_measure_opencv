from ultralytics import YOLO
import cv2, os

MODEL   = "yolov8l.pt"
IMAGE   = "data/cat.png"
CONF    = 0.5
PADDING = 10
OUT_DIR = "crops"


def load_model(path=MODEL):
    return YOLO(path)


def detect(model, image_path, conf=CONF):
    return model(image_path, conf=conf)[0]


def crop(image_path, boxes, model_names, padding=PADDING, out_dir=OUT_DIR):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    counts = {}

    for box in boxes:
        name = model_names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1 = max(0, int(x1) - padding), max(0, int(y1) - padding)
        x2, y2 = min(w, int(x2) + padding), min(h, int(y2) + padding)
        fname = f"{base}_{name}_{counts[name]}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), img[y1:y2, x1:x2])
        print(f"saved: {fname}")

def cropped(img, model=None, padding=PADDING, conf=CONF, filter_names=("cat", "dog")):
    if model is None:
        model = load_model()
    results = detect(model, img, conf=conf)
    h, w = img.shape[:2]
    found = []

    for box in results.boxes:
        name = model.names[int(box.cls[0])]
        if not any(f in name for f in filter_names):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1 = max(0, int(x1) - padding), max(0, int(y1) - padding)
        x2, y2 = min(w, int(x2) + padding), min(h, int(y2) + padding)
        found.append([x1, y1, x2, y2])

    return found  # [] if nothing matched

def run(image_path=IMAGE, conf=CONF):
    model   = load_model()
    results = detect(model, image_path, conf=conf)
    crop(image_path, results.boxes, model.names)

if __name__ == "__main__":
    run()