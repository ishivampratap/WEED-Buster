from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Automated Weed Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}


def classify_single_region(original_rgb, original_hsv, raw_mask_roi):
    """
    Classify a single plant/leaf region using multiple features.
    
    WEEDS (grass): thin, elongated, lighter yellow-green, many fine edges
    CROPS (guava): broad, round, deeper green, smoother surface
    """
    green_pixels = raw_mask_roi > 0
    green_count = int(np.sum(green_pixels))
    if green_count < 50:
        return False, 0.5, {}
    
    score = 0.0
    
    # --- LEAF SHAPE: aspect ratio of the entire region's contour ---
    contours_inner, _ = cv2.findContours(raw_mask_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Weighted average aspect ratio across all sub-contours
    total_area = 0
    weighted_ar = 0
    thin_area = 0
    broad_area = 0
    for c in contours_inner:
        a = cv2.contourArea(c)
        if a < 30:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) <= 0:
            continue
        ar = max(rw, rh) / min(rw, rh)
        weighted_ar += ar * a
        total_area += a
        if ar > 2.5:
            thin_area += a
        else:
            broad_area += a
    
    avg_ar = weighted_ar / total_area if total_area > 0 else 1.0
    thin_ratio = thin_area / (thin_area + broad_area) if (thin_area + broad_area) > 0 else 0
    
    # Shape scoring
    if avg_ar > 3.0:
        score += 0.35
    elif avg_ar > 2.5:
        score += 0.20
    elif avg_ar > 2.0:
        score += 0.05
    elif avg_ar < 1.8:
        score -= 0.25
    else:
        score -= 0.10
    
    # Thin ratio boost
    if thin_ratio > 0.5:
        score += 0.20
    elif thin_ratio < 0.15:
        score -= 0.15
    
    # --- COLOR: hue and saturation ---
    h_vals = original_hsv[:, :, 0][green_pixels]
    s_vals = original_hsv[:, :, 1][green_pixels]
    avg_hue = float(np.mean(h_vals))
    avg_sat = float(np.mean(s_vals))
    
    if avg_hue < 36:
        score += 0.20
    elif avg_hue < 42:
        score += 0.08
    elif avg_hue > 50:
        score -= 0.20
    else:
        score -= 0.08
    
    if avg_sat < 60:
        score += 0.08
    elif avg_sat > 130:
        score -= 0.08
    
    # --- TEXTURE: edge density ---
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_count = np.sum((edges > 0) & green_pixels)
    edge_density = edge_count / float(green_count)
    
    if edge_density > 0.25:
        score += 0.15
    elif edge_density < 0.08:
        score -= 0.10
    
    is_weed = score > 0.0  # Lowered threshold to more readily classify thin plants as weeds
    confidence = min(0.99, 0.55 + abs(score) * 1.5)
    
    features = {
        "thin_ratio": round(thin_ratio, 2),
        "avg_aspect": round(avg_ar, 1),
        "hue": round(avg_hue, 1),
        "score": round(score, 3)
    }
    
    return is_weed, confidence, features


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        annotated_img = img_array.copy()
        img_h, img_w = img_array.shape[:2]

        # ============================================================
        #  STEP 1: GREEN SEGMENTATION
        # ============================================================
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_green = np.array([22, 20, 20])  # Lowered S and V bounds for darker leaves; broadened H for yellower greens
        upper_green = np.array([105, 255, 255])
        raw_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Remove noise
        noise_k = np.ones((5, 5), np.uint8)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, noise_k)
        
        # ============================================================
        #  STEP 2: FIND INDIVIDUAL PLANT REGIONS
        #  Use connected components on the raw mask. Each physically
        #  separate green region becomes its own detection candidate.
        #  Then group only those that are VERY close together.
        # ============================================================
        
        # Light close to connect leaves within ONE plant (~1% range)
        k1 = max(7, int(min(img_h, img_w) * 0.01))
        k1 = k1 if k1 % 2 == 1 else k1 + 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
        grouped = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # Use connected components instead of contours for cleaner separation
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grouped)
        
        # Minimum area: 0.05% of image (very small to catch individual thin grass blades)
        min_area = max(100, int((img_h * img_w) * 0.0005))
        
        # Collect candidate regions
        candidates = []  # (x1, y1, x2, y2, centroid_x, centroid_y)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            candidates.append([x, y, x + w, y + h, cx, cy, area])
        
        # ============================================================
        #  STEP 3: MERGE ONLY VERY CLOSE REGIONS (same plant)
        #  Use centroid distance: merge if centroids are < 5% of image apart
        # ============================================================
        merge_dist = min(img_h, img_w) * 0.05
        
        # Build clusters using centroid proximity
        clusters = []
        assigned = [False] * len(candidates)
        
        for i in range(len(candidates)):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            # Find all candidates close to this one (BFS)
            queue = [i]
            while queue:
                current = queue.pop(0)
                cx1, cy1 = candidates[current][4], candidates[current][5]
                for j in range(len(candidates)):
                    if assigned[j]:
                        continue
                    cx2, cy2 = candidates[j][4], candidates[j][5]
                    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                    if dist < merge_dist:
                        cluster.append(j)
                        assigned[j] = True
                        queue.append(j)
            clusters.append(cluster)
        
        # Build merged bounding boxes per cluster
        plant_bboxes = []
        for cluster in clusters:
            x1 = min(candidates[i][0] for i in cluster)
            y1 = min(candidates[i][1] for i in cluster)
            x2 = max(candidates[i][2] for i in cluster)
            y2 = max(candidates[i][3] for i in cluster)
            total_area = sum(candidates[i][6] for i in cluster)
            plant_bboxes.append((x1, y1, x2, y2, total_area))
        
        # ============================================================
        #  STEP 4: CLASSIFY EACH PLANT
        # ============================================================
        detections = []
        crop_count = 0
        weed_count = 0
        
        for (x1, y1, x2, y2, plant_area) in plant_bboxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            roi_rgb = img_array[y1:y2, x1:x2]
            roi_hsv = hsv[y1:y2, x1:x2]
            roi_mask = raw_mask[y1:y2, x1:x2]
            
            is_weed, confidence, features = classify_single_region(roi_rgb, roi_hsv, roi_mask)
            
            if is_weed:
                label = "Weed"
                color = (255, 51, 51)
                weed_count += 1
            else:
                label = "Crop"
                color = (76, 175, 80)
                crop_count += 1
            
            # Draw bounding box
            thickness = max(2, int(min(img_h, img_w) * 0.003))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            font_scale = max(0.6, min(img_h, img_w) * 0.0006)
            text = f"{label} {confidence:.0%}"
            (tw, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(annotated_img, (x1, y1 - th_t - 12), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated_img, text, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            
            detections.append({
                "class": label,
                "confidence": float(confidence),
                "features": features
            })
        
        # Overall status
        if len(detections) > 0:
            avg_greenness = float(np.mean(raw_mask) / 255.0)
            overall_prediction = "Weed Detected" if weed_count > 0 else "All Clear (Crop Only)"
        else:
            avg_greenness = 0.0
            overall_prediction = "No Plants Found"

        annotated_pil = Image.fromarray(annotated_img)
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_img = f"data:image/jpeg;base64,{img_str}"

        return {
            "prediction": overall_prediction,
            "crop_count": crop_count,
            "weed_count": weed_count,
            "detections": detections,
            "greenness": avg_greenness,
            "size": [image.size[0], image.size[1]],
            "annotated_image": base64_img
        }
        
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}

# For local development
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")
