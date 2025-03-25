import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading

from processing.mask_generate import process_and_save_images
from processing.color_split import color_segment
from processing.line_draft import generate_sketch
from processing.combine_psd import create_psd_from_folder


def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    image_path_var.set(file_path)


def print_log(message):
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, message + "\n")
    log_text.yview(tk.END)
    log_text.config(state=tk.DISABLED)


def update_progress(current, total):
    progress_value = (current / total) * 100
    progress_bar['value'] = progress_value
    print_log(f"Progress: {current}/{total}")


def run_processing_thread():
    image_path = image_path_var.get()
    if not image_path:
        messagebox.showerror("Error", "Please select an input image.")
        return

    try:
        filename_with_extension = os.path.basename(image_path)
        filename_without_extension = os.path.splitext(filename_with_extension)[0]
        output_path = os.path.dirname(image_path)

        masked_dir = os.path.join(output_path, filename_without_extension, "masked")
        os.makedirs(masked_dir, exist_ok=True)

        line_draft_output_path = os.path.join(output_path, filename_without_extension, "masked", "9999.png")
        print_log("Generating line art...")
        sketch_none = generate_sketch(image_path, line_draft_output_path)

        print_log("Starting color block segmentation...")
        color_segment(image_path, sketch_none, output_path, print_log, max_images=max_images_var.get())

        mask_input_folder = os.path.join(output_path, filename_without_extension, "source")
        mask_output_folder = os.path.join(output_path, filename_without_extension, "masked")
        print_log("Generating masks...")
        process_and_save_images(
            mask_input_folder,
            mask_output_folder,
            min_area=min_area_var.get(),
            kernel_size=kernel_size_var.get(),
            max_workers=max_workers_var.get(),
            scale_factor=scale_factor_var.get(),
            print_log=print_log,
            progress_callback=update_progress
        )

        output_psd_path = os.path.join(output_path, filename_without_extension, "final.psd")
        print_log("Creating PSD file...")
        create_psd_from_folder(mask_output_folder, output_psd_path, print_log=print_log,
                               progress_callback=update_progress)

        messagebox.showinfo("Success", "Processing completed!")
        print_log("Processing completed!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print_log(f"Error: {str(e)}")


def run_process():
    threading.Thread(target=run_processing_thread).start()


def update_label_value(slider, label):
    label.delete(0, tk.END)
    label.insert(0, str(int(slider.get())))


window = tk.Tk()
window.title("Deep-Fake")
window.geometry("950x800")

style = ttk.Style()
style.configure("TButton", font=("Microsoft YaHei", 12), padding=10)
style.configure("TLabel", font=("Microsoft YaHei", 12))
style.configure("TEntry", font=("Microsoft YaHei", 12), padding=10)

image_path_var = tk.StringVar()
max_images_var = tk.IntVar(value=900)
min_area_var = tk.IntVar(value=5000)
kernel_size_var = tk.IntVar(value=3)
max_workers_var = tk.IntVar(value=12)
scale_factor_var = tk.IntVar(value=8)

ttk.Label(window, text="Image Path:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
entry_image = ttk.Entry(window, textvariable=image_path_var, width=40)
entry_image.grid(row=0, column=1, padx=10, pady=5)
ttk.Button(window, text="Select Image", command=browse_image).grid(row=0, column=2, padx=10, pady=5)

ttk.Label(window, text="Max Images (Color Segmentation):").grid(row=1, column=0, padx=10, pady=5, sticky='w')
max_images_slider = ttk.Scale(
    window,
    from_=1, to=500,
    variable=max_images_var,
    orient='horizontal',
    command=lambda e: update_label_value(max_images_slider, max_images_label)
)
max_images_slider.grid(row=1, column=1, padx=10, pady=5, columnspan=2)
max_images_label = ttk.Entry(window, textvariable=max_images_var, width=5)
max_images_label.grid(row=1, column=3, padx=10, pady=5)
max_images_label.bind("<FocusOut>", lambda e: max_images_var.set(int(max_images_label.get())))

ttk.Label(window, text="Min Area\n(Mask minimum detection area):").grid(row=2, column=0, padx=10, pady=5, sticky='w')
min_area_slider = ttk.Scale(
    window,
    from_=10, to=100000,
    variable=min_area_var,
    orient='horizontal',
    command=lambda e: update_label_value(min_area_slider, min_area_label)
)
min_area_slider.grid(row=2, column=1, padx=10, pady=5, columnspan=2)
min_area_label = ttk.Entry(window, textvariable=min_area_var, width=5)
min_area_label.grid(row=2, column=3, padx=10, pady=5)
min_area_label.bind("<FocusOut>", lambda e: min_area_var.set(int(min_area_label.get())))

ttk.Label(window, text="Kernel Size\n(Color block expansion range):").grid(row=3, column=0, padx=10, pady=5, sticky='w')
kernel_size_slider = ttk.Scale(
    window,
    from_=1, to=10,
    variable=kernel_size_var,
    orient='horizontal',
    command=lambda e: update_label_value(kernel_size_slider, kernel_size_label)
)
kernel_size_slider.grid(row=3, column=1, padx=10, pady=5, columnspan=2)
kernel_size_label = ttk.Entry(window, textvariable=kernel_size_var, width=5)
kernel_size_label.grid(row=3, column=3, padx=10, pady=5)
kernel_size_label.bind("<FocusOut>", lambda e: kernel_size_var.set(int(kernel_size_label.get())))

ttk.Label(window, text="Max Workers\n(Parallel threads, adjust based on hardware):").grid(
    row=4, column=0, padx=10, pady=5, sticky='w'
)
max_workers_slider = ttk.Scale(
    window,
    from_=1, to=20,
    variable=max_workers_var,
    orient='horizontal',
    command=lambda e: update_label_value(max_workers_slider, max_workers_label)
)
max_workers_slider.grid(row=4, column=1, padx=10, pady=5, columnspan=2)
max_workers_label = ttk.Entry(window, textvariable=max_workers_var, width=5)
max_workers_label.grid(row=4, column=3, padx=10, pady=5)
max_workers_label.bind("<FocusOut>", lambda e: max_workers_var.set(int(max_workers_label.get())))

ttk.Label(window, text="Scale Factor (Fineness of scaling):").grid(row=5, column=0, padx=10, pady=5, sticky='w')
scale_factor_slider = ttk.Scale(
    window,
    from_=1, to=20,
    variable=scale_factor_var,
    orient='horizontal',
    command=lambda e: update_label_value(scale_factor_slider, scale_factor_label)
)
scale_factor_slider.grid(row=5, column=1, padx=10, pady=5, columnspan=2)
scale_factor_label = ttk.Entry(window, textvariable=scale_factor_var, width=5)
scale_factor_label.grid(row=5, column=3, padx=10, pady=5)
scale_factor_label.bind("<FocusOut>", lambda e: scale_factor_var.set(int(scale_factor_label.get())))

ttk.Label(window, text="Processing Progress:").grid(row=6, column=0, padx=10, pady=5, sticky='w')
progress_bar = ttk.Progressbar(window, orient='horizontal', length=400, mode='determinate')
progress_bar.grid(row=6, column=1, padx=10, pady=5, columnspan=2)

log_text = ScrolledText(window, width=70, height=15, font=("Microsoft YaHei", 10), wrap=tk.WORD)
log_text.grid(row=7, column=0, columnspan=3, padx=10, pady=10)
log_text.config(state=tk.DISABLED)

ttk.Button(window, text="Run Processing", command=run_process).grid(row=8, column=1, padx=10, pady=20)

if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    window.mainloop()
