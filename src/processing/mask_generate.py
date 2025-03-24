import cv2
import numpy as np
from PIL import Image
from collections import Counter
import os
import concurrent.futures


def process_image(image_path, min_area=100, kernel_size=3, scale_factor=4):
    image = cv2.imread(image_path)
    original_size = (image.shape[1], image.shape[0])
    # Upscale image
    upscaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(upscaled)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    edges = cv2.Canny(mask[:, :, 0], 150, 200)
    kernel_mat = np.ones((kernel_size, kernel_size), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_mat)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    for contour in contours:
        epsilon = 0.0015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(final_mask, [approx], -1, (255, 255, 255), thickness=cv2.FILLED)

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_mat)
    mask_dilated = cv2.dilate(mask, kernel_mat, iterations=1)
    mask_eroded = cv2.erode(mask_dilated, kernel_mat, iterations=1)
    combined_mask = cv2.bitwise_or(final_mask, mask_eroded)

    # Downscale mask back to original size
    combined_mask = cv2.resize(combined_mask, original_size, interpolation=cv2.INTER_AREA)
    return combined_mask


def get_most_common_color(image_path):
    image = Image.open(image_path).convert("RGBA")
    pixels = list(image.getdata())
    non_transparent = [pixel[:3] for pixel in pixels if pixel[3] > 0]
    most_common = Counter(non_transparent).most_common(1)[0][0]
    width, height = image.size
    color_block = np.zeros((height, width, 3), dtype=np.uint8)
    color_block[:, :] = most_common[::-1]  # BGR order for OpenCV
    return color_block, most_common


def process_single_image(image_path, output_folder, min_area, kernel_size, scale_factor=2):
    try:
        color_block, _ = get_most_common_color(image_path)
        combined_mask = process_image(image_path, min_area, kernel_size, scale_factor)
        # Convert mask to binary to ensure pixels are either 255 or 0
        mask_gray = combined_mask[:, :, 0]
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        # Use binary mask to select either color_block or black
        result = np.where(binary_mask[:, :, None] == 255, color_block, 0).astype(np.uint8)
        if cv2.countNonZero(binary_mask) > 0:
            result_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            result_with_alpha[:, :, 3] = binary_mask
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask.png")
            cv2.imwrite(output_path, result_with_alpha)
        return f"Processed: {os.path.basename(image_path)}"
    except Exception as e:
        return f"Error processing {os.path.basename(image_path)}: {str(e)}"


def process_and_save_images(input_folder, output_folder, min_area=100, kernel_size=3, scale_factor=2, max_workers=5,
                            print_log=None, progress_callback=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    total = len(image_files)
    futures_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for filename in image_files:
            image_path = os.path.join(input_folder, filename)
            futures_list.append(
                executor.submit(process_single_image, image_path, output_folder, min_area, kernel_size, scale_factor))
        for idx, future in enumerate(concurrent.futures.as_completed(futures_list), start=1):
            message = future.result()
            if progress_callback:
                progress_callback(idx, total)
            if print_log:
                print_log(message)
