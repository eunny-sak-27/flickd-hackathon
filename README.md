# Flickd Fashion AI Backend

This project powers the backend system for a Gen Z–focused fashion discovery experience. It processes short videos and generates structured JSON output containing detected fashion products and the overall aesthetic “vibe.”

## 🔍 What It Does

Given a short video and its transcript, this pipeline:
- Detects fashion items using a custom Roboflow object detection model
- Matches detected items to a product catalog using CLIP embeddings and FAISS vector search
- Classifies the "vibe" (e.g., Y2K, Coquette) based on the video transcript
- Outputs a final JSON file with matched products and predicted vibes

---


## What the System Does

1. **Input:** Short video + (optional) transcript or caption  
2. **Output:** One JSON per video with:
- `video_id`
- Top 1–3 fashion **vibes** (e.g., "Coquette", "Party Glam")
- Matched **products** with:
  - Type (e.g., dress, top)
  - Color
  - Match type (exact, similar, no match)
  - Confidence
  - Matched product ID


---

## Tech Stack

| Task                     | Tool Used                            |
|--------------------------|---------------------------------------|
| Object Detection         | YOLOv8 via Roboflow                   |
| Image Embeddings         | CLIP (ViT-B/32 and ViT-L/14)          |
| Similarity Search        | FAISS (Product catalog indexing)      |
| Vibe Classification      | BART-Large-MNLI (Zero-Shot pipeline)  |
| Transcript Input (optional) | Manual / Whisper / AssemblyAI     |
| Code Environment         | Python, OpenCV, Pandas, Transformers  |

---

## 🧠 How CLIP + FAISS Was Built

The product catalog contains >1,000 images of clothing items. To enable fast matching:

1. **Image Embeddings**: Each product image is passed through the CLIP (ViT-L/14) model to obtain a 768D embedding vector.
2. **Vector Indexing**: These embeddings are stored in a FAISS index for efficient nearest-neighbor search.
3. **Similarity Search**: At runtime, detected item crops are encoded with CLIP and matched against the FAISS index to retrieve the most visually similar product.

---
## 🧪 Example Output

```json
{
  "video_id": "abc123",
  "vibes": ["Coquette", "Evening"],
  "products": [
    {
      "type": "dress",
      "color": "black",
      "match_type": "similar",
      "matched_product_id": "prod_456",
      "confidence": 0.84
    }
  ]
}
```
---
 ## How It Works (Pipeline)
 
- Detects fashion items using a custom Roboflow object detection model
- Detect Clothes in frames using Roboflow (YOLOv8 model)
- Crop Objects and embed them using CLIP (ViT-B/32)
- Compare with Catalog using FAISS similarity search
- Classify Color using CLIP and softmax against color labels
- Deduplicate detections by type + product ID
- Classify Vibes using BART-MNLI (Zero-shot inference on transcript)
- Output final JSON with deduped product and top 3 vibe labels


---
## 🛠 Challenges & Iterations

| Challenge                                                               | Solution / Iteration                                                                 |
|------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Initial output had multiple JSONs per frame                            | Refactored logic to generate a single consolidated JSON output per video            |
| Slow runtime using ViT-L/14 CLIP model                                 | Switched to ViT-B/32 for faster inference with minimal loss in performance          |
| Cropped frames lost visual context                                     | Added padding around bounding boxes before cropping for better visual features      |
| Duplicate product matches in output                                    | Implemented deduplication based on (type, matched_product_id) keys                  |
| Vibe classifier returned generic results with DistilBERT               | Replaced with BART-MNLI zero-shot classifier for more accurate vibe predictions     |
| Original product catalog had only 1 image per item                     | Augmented by sampling 2 random images per product to improve matching consistency   |
| FAISS index mismatch due to model inconsistency                        | Rebuilt the index with embeddings generated from ViT-B/32 for consistency           |
---
## 📦 How to Run

- Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
- Input the video and text/transcript links
- Ensure you have the following files in place:

      -test_vid_1.mp4 — input video
      -test_text_1.txt — video transcript
      -faiss_product_index.idx — FAISS index with product embeddings

- product_ids.csv — mapping of index to product IDs

- Run model.py
---
## Limitations:
- Matching accuracy depends on Roboflow detection quality and catalog coverage.
- Vibe classification is based only on the text file provided, not video visuals.
- Low-light or fast-motion videos may reduce detection performance.
---

## Built By
- Saketh Eunny
- Mail: saketheunny1@gmail.com
- [LinkedIn](https://www.linkedin.com/in/saketh-eunny-a7b9b2231)
