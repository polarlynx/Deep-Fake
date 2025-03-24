import cv2
import numpy as np
import os

def color_segment(image_path, img, output_dir, print_log, max_images=200, hue_step=10, sat_step=40, val_step=40):
    if img is None:
        raise ValueError("Image not found or invalid image path")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get the image file name (without path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create the output folder
    output_folder = os.path.join(output_dir, image_name, "source")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract the minimum and maximum HSV values
    h_min, s_min, v_min = np.min(img_hsv, axis=(0, 1))
    h_max, s_max, v_max = np.max(img_hsv, axis=(0, 1))

    # Calculate the number of segments and adjust step size
    def calculate_segments():
        h_segments = (h_max - h_min) // hue_step
        s_segments = (s_max - s_min) // sat_step
        v_segments = (v_max - v_min) // val_step
        return np.int64(h_segments) * np.int64(s_segments) * np.int64(v_segments)

    while True:
        total_segments = calculate_segments()
        if total_segments <= max_images:
            break
        hue_step = max(1, hue_step + 1)
        sat_step = max(1, sat_step + 5)
        val_step = max(1, val_step + 5)

    print_log(f"Final step sizes: Hue:{hue_step}, Saturation:{sat_step}, Value:{val_step}")
    print_log(f"Estimated number of segmentation intervals: {total_segments}")
    print(f"Final step sizes: Hue:{hue_step}, Saturation:{sat_step}, Value:{val_step}")
    print(f"Estimated number of segmentation intervals: {total_segments}")

    # Collect all valid segmentation results
    segments = []
    for h in range(h_min, h_max, hue_step):
        for s in range(s_min, s_max, sat_step):
            for v in range(v_min, v_max, val_step):
                lower = np.array([h, s, v])
                upper = np.array([
                    min(h + hue_step, 179),
                    min(s + sat_step, 255),
                    min(v + val_step, 255)
                ])

                mask = cv2.inRange(img_hsv, lower, upper)
                pixel_count = cv2.countNonZero(mask)

                if pixel_count > 0:
                    result = cv2.bitwise_and(img, img, mask=mask)
                    result_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
                    result_with_alpha[:, :, 3] = mask
                    segments.append((pixel_count, result_with_alpha))

    # Sort in descending order by pixel count
    segments.sort(key=lambda x: x[0], reverse=True)

    # Save the first max_images results
    saved_count = 0
    for idx, (pixel_count, segment) in enumerate(segments):
        if saved_count >= max_images:
            break
        output_path = os.path.join(output_folder, f"{idx:04d}.png")
        cv2.imwrite(output_path, segment)
        saved_count += 1

    print(f"Actual number of saved segmented images: {saved_count}")
    print_log(f"Actual number of saved segmented images: {saved_count}")
