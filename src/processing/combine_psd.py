from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from PIL import Image
import os

def create_psd_from_folder(folder_path, output_psd_path, print_log=None, progress_callback=None):
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    if not png_files:
        if print_log:
            print_log("No PNG files found.")
        return

    # Choose the size of the first image
    first_image = Image.open(os.path.join(folder_path, png_files[0]))
    img_width, img_height = first_image.size

    # Create a new PSD file with the same size as the image
    psd = PSDImage.new(
        mode='RGB',
        size=(img_width, img_height),
        color=255,  # White background
        depth=8
    )

    total_files = len(png_files)
    for idx, png_file in enumerate(png_files):
        png_image = Image.open(os.path.join(folder_path, png_file))
        layer_name = os.path.splitext(png_file)[0]

        # Convert the PNG image to PixelLayer and set the layer name
        png_layer = PixelLayer.frompil(png_image)
        png_layer.name = layer_name

        psd.append(png_layer)

        if progress_callback:
            progress_callback(idx + 1, total_files)

        if print_log:
            print_log(f"Processing image: {png_file} ({idx + 1}/{total_files})")

    if print_log:
        print_log("Saving PSD file, it may take a while. Please wait...")
    psd.save(output_psd_path)
    if print_log:
        print_log(f"PSD file saved to: {output_psd_path}")
