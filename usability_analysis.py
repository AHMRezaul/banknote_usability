import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import streamlit as st
from sklearn.cluster import DBSCAN

def remove_background(image):
    """Remove white background from image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = image.copy()
    result[mask == 0] = [255, 255, 255]
    return result, mask

def create_binary_mask(image):
    """Create binary mask for note area"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def align_images(img1, img2):
    """Align img1 (torn) to img2 (standard)"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        raise ValueError("Could not detect features for alignment")
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        raise ValueError("Not enough good matches found for alignment")
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    return aligned_img

def calculate_rgb_damage(aligned_torn, reference):
    """Calculate damage using direct RGB differences"""
    ref_mask = create_binary_mask(reference)
    torn_mask = create_binary_mask(aligned_torn)
    common_mask = cv2.bitwise_and(ref_mask, torn_mask)
    rgb_diff = cv2.absdiff(aligned_torn, reference)
    rgb_diff_mask = cv2.cvtColor(rgb_diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(rgb_diff_mask, common_mask)
    
    total_pixels = np.sum(common_mask > 0)
    if total_pixels > 0:
        avg_rgb_diff = np.sum(masked_diff) / (total_pixels * 255.0)
    else:
        avg_rgb_diff = 0.0
    
    return masked_diff, avg_rgb_diff

def detect_edge_corner_damage(aligned_torn, standard_img, overlap_threshold=0.1):
    """
    Detects edge and corner damage.
    
    Args:
        aligned_torn (np.array): Aligned torn note image.
        standard_img (np.array): Aligned standard note image (same size).
        overlap_threshold (float): % overlap threshold to consider damaged (0.1 = 10%).

    Returns:
        dict: Report of which edges and corners are damaged.
    """
    def create_binary_mask(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    standard_binary = create_binary_mask(standard_img)
    aligned_binary = create_binary_mask(aligned_torn)
    damage_mask = cv2.subtract(standard_binary, aligned_binary)

    h, w = damage_mask.shape

    edge_thickness = int(0.05 * min(h, w))  

    masks = {
        'Top Edge': np.zeros((h, w), dtype=np.uint8),
        'Bottom Edge': np.zeros((h, w), dtype=np.uint8),
        'Left Edge': np.zeros((h, w), dtype=np.uint8),
        'Right Edge': np.zeros((h, w), dtype=np.uint8),
        'Top Left Corner': np.zeros((h, w), dtype=np.uint8),
        'Top Right Corner': np.zeros((h, w), dtype=np.uint8),
        'Bottom Left Corner': np.zeros((h, w), dtype=np.uint8),
        'Bottom Right Corner': np.zeros((h, w), dtype=np.uint8),
    }

    masks['Top Edge'][:edge_thickness, :] = 255
    masks['Bottom Edge'][-edge_thickness:, :] = 255
    masks['Left Edge'][:, :edge_thickness] = 255
    masks['Right Edge'][:, -edge_thickness:] = 255

    corner_size_x = int(0.1 * w)
    corner_size_y = int(0.1 * h)

    masks['Top Left Corner'][:corner_size_y, :corner_size_x] = 255
    masks['Top Right Corner'][:corner_size_y, -corner_size_x:] = 255
    masks['Bottom Left Corner'][-corner_size_y:, :corner_size_x] = 255
    masks['Bottom Right Corner'][-corner_size_y:, -corner_size_x:] = 255

    report = {}
    for region, mask in masks.items():
        region_mask = cv2.bitwise_and(damage_mask, damage_mask, mask=mask)
        total_area = np.sum(mask > 0)
        damaged_area = np.sum(region_mask > 0)
        overlap_ratio = damaged_area / total_area if total_area > 0 else 0

        report[region] = {
            'overlap_percentage': overlap_ratio * 100,
            'is_damaged': overlap_ratio >= overlap_threshold
        }

    print("\nEdge & Corner Damage Report:")
    for region, data in report.items():
        status = "Damaged" if data['is_damaged'] else "OK"
        print(f"{region}: {status} ({data['overlap_percentage']:.1f}% damaged)")

    return report


def apply_gray_world_white_balance(img):
    """Basic gray-world white balance correction."""
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] *= (avg_gray / avg_b)
    result[:, :, 1] *= (avg_gray / avg_g)
    result[:, :, 2] *= (avg_gray / avg_r)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# def color_distortion_scoring(aligned_torn, standard_img, show_heatmap=True):
#     """
#     Measure color distortion between aligned_torn and standard_img using ΔE (LAB space).
    
#     Args:
#         aligned_torn (np.array): Aligned torn note.
#         standard_img (np.array): Pristine standard note.
#         show_heatmap (bool): Whether to display ΔE heatmap.
    
#     Returns:
#         dict: Mean and max color distortion scores.
#     """
#     # Step 1: Apply white balance correction
#     wb_torn = apply_gray_world_white_balance(aligned_torn)
#     wb_standard = apply_gray_world_white_balance(standard_img)

#     # Step 2: Convert to LAB
#     lab_torn = cv2.cvtColor(wb_torn, cv2.COLOR_BGR2LAB)
#     lab_standard = cv2.cvtColor(wb_standard, cv2.COLOR_BGR2LAB)

#     # Step 3: Convert LAB images to float (scikit-image expects float [0, 100], [-128, 127])
#     lab_torn_float = lab_torn.astype(np.float32)
#     lab_standard_float = lab_standard.astype(np.float32)

#     # Scale L channel to 0-100 and A/B to -128 to 127
#     lab_torn_float[:, :, 0] = lab_torn_float[:, :, 0] * (100.0 / 255.0)
#     lab_torn_float[:, :, 1:] = lab_torn_float[:, :, 1:] - 128.0

#     lab_standard_float[:, :, 0] = lab_standard_float[:, :, 0] * (100.0 / 255.0)
#     lab_standard_float[:, :, 1:] = lab_standard_float[:, :, 1:] - 128.0

#     # Step 4: Compute ΔE (CIE76)
#     delta_e = color.deltaE_cie76(lab_standard_float, lab_torn_float)

#     # Step 5: Mask to only note area (from standard image)
#     binary_mask = create_binary_mask(standard_img)
#     note_area = binary_mask > 0

#     mean_delta_e = np.mean(delta_e[note_area])
#     max_delta_e = np.max(delta_e[note_area])

#     print("\nColor Distortion Report:")
#     print(f"Mean ΔE (color distortion): {mean_delta_e:.2f}")
#     print(f"Max ΔE (color distortion): {max_delta_e:.2f}")

#     # Optional: Show heatmap
#     if show_heatmap:
#         plt.figure(figsize=(8, 6))
#         plt.imshow(delta_e, cmap='inferno')
#         plt.title(f'Color Distortion Heatmap (ΔE)\nMean: {mean_delta_e:.2f} | Max: {max_delta_e:.2f}')
#         plt.axis('off')
#         plt.colorbar(label='ΔE')
#         plt.tight_layout()
#         # plt.show()
#         st.pyplot(plt)

#     return {
#         'mean_delta_e': mean_delta_e,
#         'max_delta_e': max_delta_e
#     }

# Helper: Binary mask function reused
def create_binary_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def advanced_feature_detection_and_check(aligned_torn, standard_img, min_match_score=0.5):
    """
    Detect key features (denomination, portrait, seal, serials, signature) from standard note,
    match them against torn note, and mark presence.
    """
    print("\n Advanced feature detection & checking starting...")

    gray_standard = cv2.cvtColor(standard_img, cv2.COLOR_BGR2GRAY)
    gray_torn = cv2.cvtColor(aligned_torn, cv2.COLOR_BGR2GRAY)

    h, w = gray_standard.shape

    thresh = cv2.adaptiveThreshold(gray_standard, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"➡ Detected {len(contours)} raw contours.")

    contour_info = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour_info.append({'contour': cnt, 'area': area, 'centroid': (cX, cY)})

    print(f"➡ {len(contour_info)} contours kept after filtering.")

    if not contour_info:
        print("No valid contours found.")
        return {}

    coords = np.array([info['centroid'] for info in contour_info])
    clustering = DBSCAN(eps=20, min_samples=1).fit(coords)
    labels = clustering.labels_

    num_clusters = len(set(labels))
    print(f"Grouped into {num_clusters} feature clusters.")

    feature_results = {}
    vis_img = aligned_torn.copy()

    for cluster_id in range(num_clusters):
        cluster_contours = [info['contour'] for i, info in enumerate(contour_info) if labels[i] == cluster_id]
        if not cluster_contours:
            continue

        # Merge all contours in this cluster
        merged_mask = np.zeros_like(gray_standard)
        cv2.drawContours(merged_mask, cluster_contours, -1, 255, -1)

        # Get bounding box of merged region
        x, y, bw, bh = cv2.boundingRect(np.vstack(cluster_contours))

        if bw < 10 or bh < 10:
            continue  # skip trivial

        template = gray_standard[y:y+bh, x:x+bw]

        res = cv2.matchTemplate(gray_torn, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        status = "Present" if max_val >= min_match_score else "Missing/Partial"

        feature_name = f"Feature Cluster {cluster_id + 1}"
        feature_results[feature_name] = {
            'bbox_in_standard': (x, y, bw, bh),
            'match_score': max_val,
            'status': status
        }


        color = (0, 255, 0) if status == "Present" else (0, 0, 255)
        cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), color, 2)
        

    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title('Advanced Feature Detection & Checking')
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    st.pyplot(plt)

    return feature_results




def analyze_currency(torn_path, standard_path):
    """Main analysis function (no saving, reports top 5 damaged areas)"""
    # Read and process images
    torn_img = cv2.imread(torn_path)
    standard_img = cv2.imread(standard_path)
    
    if torn_img is None or standard_img is None:
        raise ValueError("Could not read images")
    
    # Resize images
    size = (800, 400)
    torn_img = cv2.resize(torn_img, size)
    standard_img = cv2.resize(standard_img, size)
    
    # Process images
    print("Removing backgrounds...")
    torn_clean, _ = remove_background(torn_img)
    standard_clean, _ = remove_background(standard_img)
    
    print("Aligning images...")
    aligned_torn = align_images(torn_clean, standard_clean)
    
    print("Processing analysis...")
    standard_binary = create_binary_mask(standard_clean)
    aligned_binary = create_binary_mask(aligned_torn)
    binary_diff = cv2.subtract(standard_binary, aligned_binary)

    print("Detecting edge and corner damage...")
    edge_corner_report = detect_edge_corner_damage(aligned_torn, standard_clean)

    # print("Calculating color distortion...")
    # color_distortion_report = color_distortion_scoring(aligned_torn, standard_clean, show_heatmap=False)

    print("Checking key features presence...")
    feature_report = advanced_feature_detection_and_check(aligned_torn, standard_clean)
    
    # Calculate binary damage ratio
    total_pixels = np.sum(standard_binary > 0)
    missing_pixels = np.sum(binary_diff > 0)
    binary_damage_ratio = (missing_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Extract connected components (for top 5 damaged areas)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_diff, connectivity=8)
    
    damage_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:  # Filter small noise
            cx, cy = centroids[i]
            damage_areas.append({
                'area': area,
                'centroid': (int(cx), int(cy))
            })
    
    # Sort by area (largest first)
    damage_areas = sorted(damage_areas, key=lambda x: x['area'], reverse=True)
    
    height, width = aligned_torn.shape[:2]
    def get_zone(x, y, w, h):
        """Map centroid to a broad zone."""
        row = ''
        col = ''
        if y < h / 3:
            row = 'Top'
        elif y < (2 * h) / 3:
            row = 'Middle'
        else:
            row = 'Bottom'
        if x < w / 3:
            col = 'Left'
        elif x < (2 * w) / 3:
            col = 'Center'
        else:
            col = 'Right'
        return f"{row} {col}"
    
    print("\nTop 5 damaged areas (if available):")
    for idx, region in enumerate(damage_areas[:5], 1):
        cx, cy = region['centroid']
        zone = get_zone(cx, cy, width, height)
        print(f"Damage {idx}: Location - {zone}, Area: {region['area']} px")
    
    # RGB analysis
    rgb_diff_mask, rgb_damage_ratio = calculate_rgb_damage(aligned_torn, standard_clean)
    rgb_damage_percentage = rgb_damage_ratio * 100
    
    # Create visualization with red highlights for damaged areas (NO CHANGES HERE)
    damage_overlay = aligned_torn.copy()
    damage_overlay[binary_diff > 0] = [0, 0, 255]  # Red color for missing areas
    
    # Display plots
    plt.figure(figsize=(14, 6))
    
    # Damage Highlighted
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(damage_overlay, cv2.COLOR_BGR2RGB))
    plt.title(f'Area Analysis (Torn ratio)\n{binary_damage_ratio:.1f}%')
    plt.axis('off')
    
    # RGB Difference
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_diff_mask, cmap='hot')
    plt.title(f'RGB Difference Analysis\n{rgb_damage_percentage:.1f}%')
    plt.axis('off')
    
    plt.tight_layout()
    # plt.show()
    st.pyplot(plt)
    
    # Print results
    print("\nDamage Analysis Results:")
    print(f"Area Damaged: {binary_damage_ratio:.1f}%")
    print(f"Color Distortion: {rgb_damage_percentage:.1f}%")

    return {
    'binary_damage': binary_damage_ratio,
    'rgb_damage': rgb_damage_percentage,
    'edge_corner_report': edge_corner_report,
    'top_damaged_areas': [
        {
            'zone': get_zone(region['centroid'][0], region['centroid'][1], width, height),
            'area_px': region['area']
        }
        for region in damage_areas[:5]
    ] if damage_areas else [],
    'feature_report': feature_report
}

