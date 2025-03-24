import os
import cv2
import numpy as np

def read_and_gray(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def generate_sketch_effect(gray):
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return sketch

def create_line_art_mask(sketch, threshold=240):
    _, mask = cv2.threshold(sketch, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask

def process_color_regions(img, mask, threshold_std=15):
    output = img.copy()
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 255 and not visited[i, j]:
                region = []
                queue = [(i, j)]
                visited[i, j] = True
                while queue:
                    x, y = queue.pop(0)
                    region.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if mask[nx, ny] == 255 and not visited[nx, ny]:
                                visited[nx, ny] = True
                                queue.append((nx, ny))
                region_colors = np.array([img[x, y] for x, y in region])
                if np.mean(np.std(region_colors, axis=0)) < threshold_std:
                    median_color = np.median(region_colors, axis=0).astype(np.uint8)
                    for x, y in region:
                        output[x, y] = median_color
    return output

def build_alpha_channel(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha = np.zeros_like(gray, dtype=np.uint8)
    alpha[mask == 255] = 255
    output_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    output_bgra[:, :, 3] = alpha
    return output_bgra

def subtract_line_art(img, mask):
    line_art_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = cv2.subtract(img, line_art_bgr)
    return result

def clean_border(image, mask, invert=False):
    h, w = mask.shape
    # Decide how to calculate the opaque_mask based on the 'invert' parameter
    if not invert:
        opaque_mask = np.where(mask == 0, 255, 0).astype(np.uint8)
    else:
        opaque_mask = np.where(mask == 255, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    inner_mask = cv2.erode(opaque_mask, kernel, iterations=2)
    border_mask = cv2.subtract(opaque_mask, inner_mask)
    result_clean = image.copy()

    for i in range(h):
        for j in range(w):
            if border_mask[i, j] > 0:
                r0, r1 = max(i - 2, 0), min(i + 3, h)
                c0, c1 = max(j - 2, 0), min(j + 3, w)
                window_inner = inner_mask[r0:r1, c0:c1]
                if np.count_nonzero(window_inner) > 0:
                    window_colors = result_clean[r0:r1, c0:c1, :]
                    valid_pixels = window_colors[window_inner > 0]
                    if valid_pixels.size > 0:
                        median_color = np.median(valid_pixels, axis=0).astype(np.uint8)
                        result_clean[i, j] = median_color

    return result_clean, kernel

def fill_gaps(result_clean, kernel):
    gray_clean = cv2.cvtColor(result_clean, cv2.COLOR_BGR2GRAY)
    _, gap_mask = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(result_clean, kernel, iterations=3)
    result_filled = result_clean.copy()
    result_filled[gap_mask == 255] = dilated[gap_mask == 255]
    return result_filled

def generate_sketch(image_path, output_path='LineArt.png'):
    img, gray = read_and_gray(image_path)
    sketch = generate_sketch_effect(gray)
    mask = create_line_art_mask(sketch)

    processed = process_color_regions(img, mask)
    # 'processed' has darker interior and lighter exterior, need to invert mask logic for cleaning
    processed_clean, _ = clean_border(processed, mask, invert=True)
    output_bgra = build_alpha_channel(processed_clean, mask)
    cv2.imwrite(output_path, output_bgra)

    result = subtract_line_art(img, mask)
    # 'result_clean' uses the normal mask logic
    result_clean, kernel = clean_border(result, mask, invert=False)
    result_filled = fill_gaps(result_clean, kernel)

    return result_filled

def split_line_art_by_component_color(line_art_path, output_paths=('thin.png', 'medium.png', 'thick.png')):
    # Read the original image, keep all channels
    orig = cv2.imread(line_art_path, cv2.IMREAD_UNCHANGED)
    if orig is None:
        raise ValueError("Image not found.")

    # Ensure the image is in BGRA format
    if orig.ndim == 3:
        if orig.shape[2] == 3:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2BGRA)
    elif orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGRA)

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(orig, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Compute connected components, treat each continuous line as one component
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    comp_thickness = {}
    thickness_values = []
    for label in range(1, num_labels):
        comp_mask = (labels == label).astype(np.uint8) * 255
        dt = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        max_dist = dt.max()  # Approximate half of the line width
        comp_thickness[label] = max_dist
        thickness_values.append(max_dist)

    if not thickness_values:
        raise ValueError("No line components found.")

    # Set classification thresholds based on the global maximum thickness
    global_max = max(thickness_values)
    T1, T2 = global_max / 3, 2 * global_max / 3

    mask_thin = np.zeros_like(gray, dtype=np.uint8)
    mask_medium = np.zeros_like(gray, dtype=np.uint8)
    mask_thick = np.zeros_like(gray, dtype=np.uint8)

    for label in range(1, num_labels):
        comp_mask = (labels == label)
        t = comp_thickness[label]
        if t < T1:
            mask_thin[comp_mask] = 255
        elif t < T2:
            mask_medium[comp_mask] = 255
        else:
            mask_thick[comp_mask] = 255

    def extract_color(mask):
        # Generate a transparent background image, keeping the original color in the line areas
        out = np.zeros_like(orig)
        out[mask == 255] = orig[mask == 255]
        return out

    out_thin = extract_color(mask_thin)
    out_medium = extract_color(mask_medium)
    out_thick = extract_color(mask_thick)

    cv2.imwrite(output_paths[0], out_thin)
    cv2.imwrite(output_paths[1], out_medium)
    cv2.imwrite(output_paths[2], out_thick)

def batch_generate_sketch(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            generate_sketch(input_path, output_path)
