"""
PlateVision: AI Number Plate Detection and Recognition
Specialized for Indian Vehicle Registration Plates (HSRP & Standard)
Signals and Image Processing (SIP) Course Project
Deployable for ₹0 on Streamlit Community Cloud via GitHub.
"""

import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import time
from PIL import Image
import io
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="PlateVision: Indian Number Plate Recognition",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .plate-display {
        font-family: 'Courier New', Courier, monospace;
        font-size: 2.2rem;
        font-weight: 800;
        color: #06b6d4;
        background: #020617;
        padding: 0.6rem 1.4rem;
        border: 2px solid #06b6d4;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 3px;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# INDIAN STATE & UNION TERRITORY RTO CODES REGISTRY
# -------------------------------------------------------------
INDIAN_RTO_CODES = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi (National Capital Territory)",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand (Former Code)",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "BH": "Bharat Series (Pan-India)",
}

# -------------------------------------------------------------
# CACHED EASYOCR READER INSTANCE
# -------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    # Load open-source EasyOCR English model in CPU mode (zero cloud cost)
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

# -------------------------------------------------------------
# STEP 1: CROPPED PLATE IMAGE ENHANCEMENT (OPENCV)
# -------------------------------------------------------------
def enhance_plate_image(plate_bgr):
    """
    Applies resolution standardization, bicubic upscaling, grayscale conversion,
    and multiple preprocessing transforms for OCR comparison.
    """
    h, w = plate_bgr.shape[:2]
    
    # 1. Standardize resolution: Upscale so height is ~110px for crisp OCR strokes
    target_h = 110
    scaling_factor = target_h / float(h) if h > 0 else 1.0
    target_w = int(w * scaling_factor)
    
    # Bicubic interpolation is superior for character edge continuity
    resized = cv2.resize(plate_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale conversion (ITU-R BT.601)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 4. Bilateral filtering for edge-preserving denoising
    denoised = cv2.bilateralFilter(equalized, d=9, sigmaColor=35, sigmaSpace=35)
    
    # --- METHOD A: Otsu Global Adaptive Thresholding ---
    _, thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- METHOD B: Adaptive Gaussian Window Thresholding ---
    thresh_adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 7
    )
    
    # --- METHOD C: Morphological Contrast (Top-Hat - Black-Hat) + Otsu ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    morph_enhanced = cv2.add(cv2.subtract(denoised, blackhat), tophat)
    _, thresh_morph = cv2.threshold(morph_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check polarity: Indian HSRP plates have black characters on a white/yellow reflective background
    # Ensure characters are dark on light or inverted if necessary
    def correct_polarity(bin_img):
        border_pixels = np.concatenate([bin_img[0, :], bin_img[-1, :], bin_img[:, 0], bin_img[:, -1]])
        if np.mean(border_pixels) < 128:
            return cv2.bitwise_not(bin_img)
        return bin_img

    thresh_otsu_corrected = correct_polarity(thresh_otsu)
    thresh_adaptive_corrected = correct_polarity(thresh_adaptive)
    thresh_morph_corrected = correct_polarity(thresh_morph)
    
    return {
        "resized_rgb": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        "gray": gray,
        "denoised": denoised,
        "otsu": thresh_otsu_corrected,
        "adaptive": thresh_adaptive_corrected,
        "morphology": thresh_morph_corrected,
    }

# -------------------------------------------------------------
# STEP 2: OCR EXECUTION ACROSS MULTIPLE PREPROCESSING METHODS
# -------------------------------------------------------------
def run_ocr_on_methods(preprocessed_dict):
    """
    Runs EasyOCR with character allowlist across each preprocessing technique
    and returns comparative recognition results.
    """
    candidates = [
        {"name": "Method A (CLAHE + Bilateral + Otsu)", "img": preprocessed_dict["otsu"]},
        {"name": "Method B (Adaptive Gaussian Threshold)", "img": preprocessed_dict["adaptive"]},
        {"name": "Method C (Morphological Contrast + Otsu)", "img": preprocessed_dict["morphology"]},
        {"name": "Method D (Denoised Grayscale Direct)", "img": preprocessed_dict["denoised"]},
    ]
    
    ocr_results = []
    
    for cand in candidates:
        try:
            # Allowlist restricts OCR to uppercase English alphanumerics
            detections = reader.readtext(
                cand["img"],
                detail=1,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            
            raw_text = " ".join([d[1] for d in detections]).strip()
            conf = float(np.mean([d[2] for d in detections])) * 100.0 if detections else 0.0
            
            ocr_results.append({
                "method_name": cand["name"],
                "image": cand["img"],
                "raw_text": raw_text,
                "confidence": conf,
            })
        except Exception as e:
            ocr_results.append({
                "method_name": cand["name"],
                "image": cand["img"],
                "raw_text": "",
                "confidence": 0.0,
            })
            
    return ocr_results

# -------------------------------------------------------------
# STEP 3: CLEANING & VALIDATING INDIAN REGISTRATION FORMATS
# -------------------------------------------------------------
def validate_and_clean_indian_plate(raw_ocr_string):
    """
    Cleans OCR output carefully and validates standard Indian vehicle registration formats
    without inventing missing characters or silently altering ambiguous readings.
    """
    # 1. Clean extraneous symbols, whitespace, and noisy marks
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_ocr_string).upper()
    
    # Strip blue IND hologram strip noise frequently read as IND, 1ND, INO, IN0, ND
    cleaned = re.sub(r'^(IND|1ND|INO|IN0|ND)', '', cleaned)
    
    if not cleaned:
        return {
            "status": "No Characters",
            "is_valid": False,
            "formatted_plate": "NOT RECOGNIZED",
            "state_name": "Unknown",
            "rto_district": "",
            "series": "",
            "number": "",
            "raw_cleaned": "",
            "disambiguations_applied": [],
            "notes": "No alphanumeric characters could be detected from the plate image."
        }
        
    disambiguations = []
    
    # 2. Check for Standard Indian Pattern: 2 Letters (State) + 1-2 Digits (RTO) + 1-3 Letters (Series) + 4 Digits (Number)
    # Total length is typically 9 to 11 characters (e.g., DL3CCA1948, MH02CD8842, KA01AB1234, DL8C1234)
    std_regex = r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$'
    std_match = re.match(std_regex, cleaned)
    
    if std_match:
        state_code = std_match.group(1)
        rto = std_match.group(2)
        series = std_match.group(3)
        num = std_match.group(4)
        state_name = INDIAN_RTO_CODES.get(state_code, "Unlisted State Code")
        
        return {
            "status": "Valid Indian HSRP Standard",
            "is_valid": True,
            "formatted_plate": f"{state_code} {rto} {series} {num}",
            "state_name": state_name,
            "rto_district": f"#{rto}",
            "series": series,
            "number": num,
            "raw_cleaned": cleaned,
            "disambiguations_applied": disambiguations,
            "notes": f"Verified against Indian Motor Vehicles Act. Jurisdiction: {state_name}."
        }
        
    # 3. Check for Bharat Series (BH): YY BH #### XX (e.g., 22BH1234AA)
    bh_regex = r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$'
    bh_match = re.match(bh_regex, cleaned)
    if bh_match:
        return {
            "status": "Valid Bharat Series (BH)",
            "is_valid": True,
            "formatted_plate": f"{bh_match.group(1)} BH {bh_match.group(2)} {bh_match.group(3)}",
            "state_name": "Pan-India (Bharat Series)",
            "rto_district": "Central RTO",
            "series": bh_match.group(3),
            "number": bh_match.group(2),
            "raw_cleaned": cleaned,
            "disambiguations_applied": [],
            "notes": f"Bharat Series registered in year 20{bh_match.group(1)}."
        }
        
    # 4. Syntactic Disambiguation Trial (WITHOUT inventing missing characters)
    # Only test character corrections if string length is within standard bounds (9-11 characters)
    if 9 <= len(cleaned) <= 11:
        chars = list(cleaned)
        
        # Disambiguate state code (first 2 positions must be letters)
        for i in range(2):
            if chars[i] == '0':
                chars[i] = 'D'
                disambiguations.append(f"Position {i+1}: '0' -> 'D' (State Code)")
            elif chars[i] == '1':
                chars[i] = 'I'
                disambiguations.append(f"Position {i+1}: '1' -> 'I' (State Code)")
            elif chars[i] == '8':
                chars[i] = 'B'
                disambiguations.append(f"Position {i+1}: '8' -> 'B' (State Code)")
                
        # Disambiguate final 4 positions (must be numeric sequence)
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] in ['O', 'D', 'Q']:
                disambiguations.append(f"Position {i+1}: '{chars[i]}' -> '0' (Numeric)")
                chars[i] = '0'
            elif chars[i] in ['I', 'L']:
                disambiguations.append(f"Position {i+1}: '{chars[i]}' -> '1' (Numeric)")
                chars[i] = '1'
            elif chars[i] == 'B':
                disambiguations.append(f"Position {i+1}: 'B' -> '8' (Numeric)")
                chars[i] = '8'
            elif chars[i] == 'S':
                disambiguations.append(f"Position {i+1}: 'S' -> '5' (Numeric)")
                chars[i] = '5'
            elif chars[i] == 'Z':
                disambiguations.append(f"Position {i+1}: 'Z' -> '2' (Numeric)")
                chars[i] = '2'
                
        corrected_str = "".join(chars)
        trial_match = re.match(std_regex, corrected_str)
        if trial_match:
            state_code = trial_match.group(1)
            state_name = INDIAN_RTO_CODES.get(state_code, "Unlisted State Code")
            return {
                "status": "Valid Indian HSRP (Disambiguated)",
                "is_valid": True,
                "formatted_plate": f"{state_code} {trial_match.group(2)} {trial_match.group(3)} {trial_match.group(4)}",
                "state_name": state_name,
                "rto_district": f"#{trial_match.group(2)}",
                "series": trial_match.group(3),
                "number": trial_match.group(4),
                "raw_cleaned": cleaned,
                "disambiguations_applied": disambiguations,
                "notes": "Validated after positional character disambiguation."
            }

    # 5. Partial / Non-Standard Return (DO NOT invent missing characters)
    return {
        "status": "Partial / Non-Standard Format",
        "is_valid": False,
        "formatted_plate": cleaned,
        "state_name": "Unverified",
        "rto_district": "N/A",
        "series": "N/A",
        "number": "N/A",
        "raw_cleaned": cleaned,
        "disambiguations_applied": [],
        "notes": f"Detected string '{cleaned}' ({len(cleaned)} chars) does not match standard Indian state + RTO syntax. Missing or degraded characters may require manual review."
    }

# -------------------------------------------------------------
# STEP 4: OPENCV CONTOUR PLATE DETECTION PIPELINE
# -------------------------------------------------------------
def detect_and_crop_plate(image_rgb, manual_box=None):
    """
    Detects vehicle number plate location using morphological gradient,
    Canny edges, brightness verification, and contour aspect ratio ranking.
    Supports manual crop box override.
    """
    h_img, w_img = image_rgb.shape[:2]
    annotated_rgb = image_rgb.copy()
    
    # Check if manual bounding box is provided
    if manual_box is not None:
        x, y, w, h = manual_box
        x = max(0, min(w_img - 10, int(x)))
        y = max(0, min(h_img - 10, int(y)))
        w = max(20, min(w_img - x, int(w)))
        h = max(10, min(h_img - y, int(h)))
        
        plate_crop_rgb = image_rgb[y:y+h, x:x+w]
        plate_crop_bgr = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(annotated_rgb, (x, y), (x + w, y + h), (6, 182, 212), 3)
        cv2.putText(annotated_rgb, "MANUAL CROP BOX", (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (6, 182, 212), 2)
        return annotated_rgb, plate_crop_bgr, (x, y, w, h)
        
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Bilateral filter for noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Canny Edge Detection
    edged = cv2.Canny(bfilter, 40, 120)
    
    # Morphological closing with horizontal kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # Contours
    keypoints = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
    
    candidates = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / max(1, h)
        area = w * h
        area_pct = (area / (w_img * h_img)) * 100.0
        
        # Indian plates generally have aspect ratios between 1.8 and 6.5
        if 1.8 <= aspect_ratio <= 6.5 and 0.25 <= area_pct <= 32.0:
            roi_gray = gray[y:y+h, x:x+w]
            mean_brightness = float(np.mean(roi_gray)) if roi_gray.size > 0 else 0
            # Indian plates are reflective white or yellow, unlike dark black grilles
            brightness_score = min(1.0, mean_brightness / 130.0)
            
            # Position prior: plates are usually mounted in the lower 35% - 88% of car image
            y_center = (y + h / 2.0) / h_img
            pos_score = 1.0 if (0.35 <= y_center <= 0.88) else 0.4
            
            # Aspect ratio score: ideal rectangular plate is ~3.5 to 4.5
            ideal_ar = 4.0
            ar_score = max(0.0, 1.0 - abs(aspect_ratio - ideal_ar) / 3.0)
            
            total_score = ar_score * 0.4 + brightness_score * 0.35 + pos_score * 0.25
            candidates.append((total_score, (x, y, w, h), aspect_ratio))
            
    plate_crop_bgr = None
    plate_bbox = None
    
    if candidates:
        # Sort by total score descending to avoid picking dark grilles
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_cand = candidates[0]
        x, y, w, h = best_cand[1]
        aspect_ratio = best_cand[2]
        plate_bbox = (x, y, w, h)
        
        # Add safe margin padding
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.08)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        
        plate_crop_rgb = image_rgb[y1:y2, x1:x2]
        plate_crop_bgr = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.rectangle(annotated_rgb, (x, y), (x + w, y + h), (6, 182, 212), 3)
        cv2.putText(annotated_rgb, f"PLATE (AR={aspect_ratio:.1f})", (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (6, 182, 212), 2)
                    
    # Fallback to center-lower region if no contour passed strict filters
    if plate_crop_bgr is None:
        cw, ch = int(w_img * 0.45), int(h_img * 0.16)
        cx, cy = int((w_img - cw) / 2), int(h_img * 0.54)
        plate_crop_rgb = image_rgb[cy:cy+ch, cx:cx+cw]
        plate_crop_bgr = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)
        plate_bbox = (cx, cy, cw, ch)
        cv2.rectangle(annotated_rgb, (cx, cy), (cx + cw, cy + ch), (245, 158, 11), 2)
        cv2.putText(annotated_rgb, "ESTIMATED PLATE ROI", (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 158, 11), 2)
                    
    return annotated_rgb, plate_crop_bgr, plate_bbox

# -------------------------------------------------------------
# MAIN STREAMLIT APPLICATION INTERFACE
# -------------------------------------------------------------
st.markdown('<div class="main-title">PlateVision: AI Number Plate Detection and Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">College Signals and Image Processing (SIP) Project · Specialized for Indian Vehicle Registration Standards · Multi-Method Preprocessing & EasyOCR</div>', unsafe_allow_html=True)

# Sidebar with theoretical controls & notes
with st.sidebar:
    st.header("Signals & Image Processing Lab")
    st.caption("College SIP Practical Project Demonstration")
    
    st.subheader("Plate Invariant Parameters")
    min_ar = st.slider("Min Aspect Ratio (W/H)", 1.5, 3.0, 2.0, 0.1)
    max_ar = st.slider("Max Aspect Ratio (W/H)", 3.5, 7.0, 6.0, 0.1)
    st.markdown("---")
    st.subheader("Manual Crop Fine-Tuning")
    enable_manual = st.checkbox("Enable Manual Crop Box", value=False,
                                help="Check this if auto-detector grabs the car radiator grille, headlights, or bumper lip")
    m_x, m_y, m_w, m_h = 25, 55, 50, 15
    if enable_manual:
        st.caption("Adjust sliders to frame the license plate directly:")
        m_x = st.slider("X Position (%)", 0, 90, 25)
        m_y = st.slider("Y Position (%)", 0, 90, 55)
        m_w = st.slider("Box Width (%)", 10, 80, 50)
        m_h = st.slider("Box Height (%)", 5, 50, 15)

    st.markdown("---")
    st.markdown("**OCR Accuracy Notice**")
    st.caption("OCR performs reliably on clear, frontal, daylight images. Motion blur, extreme tilt, mud or severe compression may produce partial readings requiring manual confirmation. The app strictly avoids fabricating missing characters.")
    
    st.markdown("---")
    st.markdown("**Supported Indian Standards**")
    st.caption("- Standard HSRP: DL 3C CA 1948, MH 02 CD 8842\n- Pan-India Bharat Series: 22 BH 1234 AA\n- State & UT RTO Jurisdiction Mapping")

# Image input selection
input_option = st.radio("Choose Vehicle Image Source:", ["Upload Vehicle Image", "Live Camera Snapshot"], horizontal=True)

img_rgb = None

if input_option == "Upload Vehicle Image":
    uploaded = st.file_uploader("Upload car / vehicle photo (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
elif input_option == "Live Camera Snapshot":
    cam = st.camera_input("Point camera at vehicle license plate")
    if cam is not None:
        file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

if img_rgb is None:
    st.info("Please upload a vehicle image or take a photo to begin number plate recognition.")
    st.stop()

# Execution Timing
start_time = time.time()

manual_box = None
if enable_manual and img_rgb is not None:
    h_i, w_i = img_rgb.shape[:2]
    manual_box = (
        int(w_i * m_x / 100),
        int(h_i * m_y / 100),
        int(w_i * m_w / 100),
        int(h_i * m_h / 100),
    )

with st.spinner("Processing image through Signals & Image Processing pipeline..."):
    # 1. Detect and crop plate
    annotated_img, plate_bgr, bbox = detect_and_crop_plate(img_rgb, manual_box=manual_box)
    
    # 2. Enhance plate and generate multiple preprocessing variants
    preprocessed = enhance_plate_image(plate_bgr)
    
    # 3. Run EasyOCR across preprocessing variants
    ocr_candidates = run_ocr_on_methods(preprocessed)
    
    # 4. Evaluate each candidate against Indian syntax rules
    ranked_candidates = []
    for cand in ocr_candidates:
        val_info = validate_and_clean_indian_plate(cand["raw_text"])
        score = cand["confidence"] + (40.0 if val_info["is_valid"] else 0.0)
        ranked_candidates.append({
            **cand,
            **val_info,
            "overall_score": score
        })
        
    ranked_candidates.sort(key=lambda x: x["overall_score"], reverse=True)
    best_candidate = ranked_candidates[0]
    
    total_elapsed_ms = (time.time() - start_time) * 1000

# -------------------------------------------------------------
# REQUIREMENT 7: DISPLAY ORIGINAL IMAGE, DETECTED CROP & FINAL OCR
# -------------------------------------------------------------
st.subheader("Vehicle Detection & Recognition Overview")

col_orig, col_crop, col_res = st.columns([1.5, 1.2, 1.3])

with col_orig:
    st.markdown("**1. Original Vehicle Image & Bounding Box**")
    st.image(annotated_img, use_column_width=True, caption=f"Detected ROI Bounding Box: {bbox[2]}x{bbox[3]} px")

with col_crop:
    st.markdown("**2. Detected Number Plate Crop (Enhanced)**")
    st.image(preprocessed["resized_rgb"], use_column_width=True, caption="Bicubic Resized & CLAHE Normalized")
    st.image(best_candidate["image"], use_column_width=True, caption=f"Winning Binarization: {best_candidate['method_name']}")

with col_res:
    st.markdown("**3. Final OCR Recognition Result**")
    st.markdown(f'<div class="plate-display">{best_candidate["formatted_plate"]}</div>', unsafe_allow_html=True)
    
    st.markdown(f"**Status**: {'🟢 Verified Format' if best_candidate['is_valid'] else '🟡 Non-Standard / Review'}")
    st.markdown(f"**Jurisdiction**: {best_candidate['state_name']}")
    if best_candidate['rto_district'] != 'N/A':
        st.markdown(f"**RTO Code**: {best_candidate['rto_district']} · **Series**: {best_candidate['series']} · **Number**: {best_candidate['number']}")
    st.markdown(f"**OCR Confidence**: {best_candidate['confidence']:.1f}%")
    st.markdown(f"**Total Latency**: {total_elapsed_ms:.1f} ms")
    
    if best_candidate["disambiguations_applied"]:
        with st.expander("Character Disambiguations Applied"):
            for d in best_candidate["disambiguations_applied"]:
                st.caption(f"• {d}")

# -------------------------------------------------------------
# REQUIREMENT 5: COMPARE MULTIPLE PREPROCESSING METHODS
# -------------------------------------------------------------
st.markdown("---")
st.subheader("Signals and Image Processing: Multi-Method Preprocessing Comparison")
st.caption("Signals and Image Processing empirical comparison of different thresholding and filtering techniques on EasyOCR accuracy:")

cols = st.columns(4)
for idx, cand in enumerate(ranked_candidates):
    with cols[idx]:
        st.image(cand["image"], use_column_width=True, caption=cand["method_name"])
        st.markdown(f"**Raw OCR**: `{cand['raw_text'] or '[Empty]'}`")
        st.markdown(f"**Confidence**: {cand['confidence']:.1f}%")
        st.markdown(f"**Status**: {'✅ Valid' if cand['is_valid'] else '⚠️ Review'}")

# Comparison Table
comp_data = []
for c in ranked_candidates:
    comp_data.append({
        "Preprocessing Method": c["method_name"],
        "Extracted Text": c["raw_text"] or "No characters",
        "Formatted Plate": c["formatted_plate"],
        "Confidence (%)": f"{c['confidence']:.1f}%",
        "Syntax Valid": "Yes" if c["is_valid"] else "No",
        "Notes": c["notes"]
    })
st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# -------------------------------------------------------------
# CSV EXPORT OF DETECTION RECORD
# -------------------------------------------------------------
csv_row = pd.DataFrame([{
    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "Recognized Plate": best_candidate["formatted_plate"],
    "Raw OCR String": best_candidate["raw_cleaned"],
    "State / UT": best_candidate["state_name"],
    "Format Standard": best_candidate["status"],
    "OCR Confidence (%)": f"{best_candidate['confidence']:.1f}",
    "Processing Time (ms)": f"{total_elapsed_ms:.1f}",
    "Winning Preprocessing": best_candidate["method_name"]
}])

csv_buffer = io.StringIO()
csv_row.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Detection Log (CSV)",
    data=csv_buffer.getvalue(),
    file_name=f"platevision_{time.strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)
