# Flickd Fashion AI Backend

This project powers the backend system for a Gen Z–focused fashion discovery experience. It processes short videos and generates structured JSON output containing detected fashion products and the overall aesthetic “vibe.”

## 🔍 What It Does

Given a short video and its transcript, this pipeline:
- Detects fashion items using a custom Roboflow object detection model
- Matches detected items to a product catalog using CLIP embeddings and FAISS vector search
- Classifies the "vibe" (e.g., Y2K, Coquette) based on the video transcript
- Outputs a final JSON file with matched products and predicted vibes

---

## ⚙️ Tech Stack

- **Object Detection:** Roboflow + custom-trained model
- **Visual Matching:** CLIP (ViT-L/14) + FAISS
- **Vibe Classification:** BART-based Zero-Shot Classifier (via HuggingFace)
- **Video Processing:** OpenCV
- **Image Preprocessing:** Pillow
- **Backend Language:** Python

---

## 🧠 How CLIP + FAISS Was Built

The product catalog contains >1,000 images of clothing items. To enable fast matching:

1. **Image Embeddings**: Each product image is passed through the CLIP (ViT-L/14) model to obtain a 768D embedding vector.
2. **Vector Indexing**: These embeddings are stored in a FAISS index for efficient nearest-neighbor search.
3. **Similarity Search**: At runtime, detected item crops are encoded with CLIP and matched against the FAISS index to retrieve the most visually similar product.

---

## 📦 How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
2. Input the video and text/transcript links
3. Ensure you have the following files in place:

4.test_vid_1.mp4 — input video

5.test_text_1.txt — video transcript

6.faiss_product_index.idx — FAISS index with product embeddings

7.product_ids.csv — mapping of index to product IDs

Run model.py

Limitations:
1.Matching accuracy depends on Roboflow detection quality and catalog coverage.
2.Vibe classification is based only on the text file provided, not video visuals.
3.Low-light or fast-motion videos may reduce detection performance.


**Sample Output:**
{
  "video_id": "2025-05-28_13-40-09_UTC",
  "vibes": [
    "Coquette",
    "Cottagecore",
    "Streetcore"
  ],
  "products": [
    {
      "type": "dress",
      "match_type": "similar",
      "matched_product_id": "42076",
      "confidence": 0.85
    }
  ]
}