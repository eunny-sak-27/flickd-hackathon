

import cv2
import numpy as np
from PIL import Image
import torch
import clip
import faiss
import pandas as pd
import json
import os
from roboflow import Roboflow
from transformers import pipeline
from collections import defaultdict, Counter

# Load Roboflow model
rf = Roboflow(api_key="XVqBcWSC02AnY4TE4Fx4")
clothing_model = rf.workspace("object-detection-bounding-box-fg9op") \
                   .project("clothing-detection-scn9m") \
                   .version(1) \
                   .model

# Load CLIP ViT-B/32
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load FAISS index and product IDs
faiss_index = faiss.read_index("/content/faiss_product_index.idx")
product_ids_df = pd.read_csv("/content/product_ids.csv")

EXACT_MATCH_THRESH = 0.9
SIMILAR_MATCH_THRESH = 0.75

def classify_match(similarity):
    if similarity > EXACT_MATCH_THRESH:
        return "exact"
    elif similarity > SIMILAR_MATCH_THRESH:
        return "similar"
    else:
        return "no match"

def pad_crop(frame, x1, y1, x2, y2, pad_percent=0.15):
    h, w, _ = frame.shape
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * pad_percent)
    pad_h = int(height * pad_percent)
    x1_p = max(0, x1 - pad_w)
    y1_p = max(0, y1 - pad_h)
    x2_p = min(w, x2 + pad_w)
    y2_p = min(h, y2 + pad_h)
    padded_crop = frame[y1_p:y2_p, x1_p:x2_p]
    return make_square(padded_crop)

def make_square(image):
    h, w, _ = image.shape
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

def get_best_match(crop_image):
    image_input = preprocess(crop_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embedding_np = embedding.cpu().numpy().astype('float32')
    D, I = faiss_index.search(embedding_np, k=1)
    similarity = float(D[0][0])
    matched_index = int(I[0][0])
    matched_product_id = product_ids_df.iloc[matched_index]['id']
    return matched_product_id, similarity, classify_match(similarity)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_vibes_from_text(text):
    candidate_vibes = [
        "Coquette", "Clean Girl", "Cottagecore", "Streetcore",
        "Y2K", "Boho", "Party Glam"
    ]
    result = classifier(text, candidate_vibes, multi_label=True)
    vibes = [label for label, score in zip(result['labels'], result['scores']) if score > 0.3]
    return vibes[:3]

def process_frame(frame):
    detections = clothing_model.predict(frame).json()
    frame_results = []
    for det in detections['predictions']:
        x, y, w, h = det['x'], det['y'], det['width'], det['height']
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        padded_crop = pad_crop(frame, x1, y1, x2, y2)
        crop_pil = Image.fromarray(cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB))
        matched_product_id, similarity, match_type = get_best_match(crop_pil)

        frame_results.append({
            "type": det['class'],
            "match_type": match_type,
            "matched_product_id": str(matched_product_id),
            "confidence": round(similarity, 2)
        })
    return frame_results

def deduplicate(products):
    grouped = defaultdict(list)
    for p in products:
        key = (p["type"], p["matched_product_id"])
        grouped[key].append(p)

    deduped = []
    for (ptype, pid), group in grouped.items():
        best = max(group, key=lambda x: x["confidence"])
        match_types = [p["match_type"] for p in group]

        deduped.append({
            "type": ptype,
            "match_type": Counter(match_types).most_common(1)[0][0],
            "matched_product_id": pid,
            "confidence": round(best["confidence"], 2)
        })
    return deduped

def main(video_path, transcript_text=None):
    cap = cv2.VideoCapture(video_path)
    all_products = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_products.extend(process_frame(frame))
    cap.release()

    all_products = convert_numpy_types(all_products)
    unique_products = deduplicate(all_products)

    if transcript_text is None:
        with open("/content/2025-05-28_13-40-09_UTC.txt", "r") as f:
            transcript_text = f.read()

    vibes = classify_vibes_from_text(transcript_text)

    final_output = {
        "video_id": os.path.basename(video_path).split(".")[0],
        "vibes": vibes,
        "products": unique_products
    }

    with open("final_output_6.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Done! Saved final_output_5.json with {len(unique_products)} products for video: {final_output['video_id']}")

if __name__ == "__main__":
    main("/content/2025-05-28_13-40-09_UTC.mp4", transcript_text="/content/2025-05-28_13-40-09_UTC.txt")


def main(video_path, transcript_text=None):
    cap = cv2.VideoCapture(video_path)
    all_products = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_products.extend(process_frame(frame))
    cap.release()

    all_products = convert_numpy_types(all_products)
    unique_products = deduplicate(all_products)

    # 🧠 STEP 1: Get most frequent type
    type_counts = Counter([p["type"] for p in unique_products])
    most_common_type = type_counts.most_common(1)[0][0]

    # 🧼 STEP 2: Filter only that type
    filtered = [p for p in unique_products if p["type"] == most_common_type]

    # ✅ STEP 3: Get highest confidence among that type
    best_product = max(filtered, key=lambda p: p["confidence"])

    # 🧠 Vibe classification
    if transcript_text is None:
        with open("/content/2025-05-28_13-40-09_UTC.txt", "r") as f:
            transcript_text = f.read()
    vibes = classify_vibes_from_text(transcript_text)

    # 📦 Final JSON
    final_output = {
        "video_id": os.path.basename(video_path).split(".")[0],
        "vibes": vibes,
        "products": [best_product]
    }

    # 💾 Save to JSON
    with open("final_output_singletype_1_2.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Saved final_output_singletype_1_2.json with product type '{most_common_type}' and matched product ID {best_product['matched_product_id']}.")

if __name__ == "__main__":
    main("/content/2025-05-28_13-40-09_UTC.mp4", transcript_text="/content/2025-05-28_13-40-09_UTC.txt")


if __name__ == "__main__":
    main("/content/2025-05-28_13-40-09_UTC.mp4", transcript_text="/content/2025-05-28_13-40-09_UTC.txt")
