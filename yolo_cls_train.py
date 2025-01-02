import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
from ultralytics import YOLO

# Load the models
od_model = YOLO('detect-best320.pt')
ic_model = YOLO('cls_best320.pt')

# Map stage labels to scores
stage_scores = {"stage1": 1, "stage2": 2, "stage3": 3, "stage4": 4}

# Global variables to track folder and image files
current_folder = ""
image_files = []
current_index = -1

def process_image(image_path):
    """Process the image and return the results and final score."""
    image = cv2.imread(image_path)

    # Run object detection
    od_results = od_model(image)
    highest_score = 0
    highest_label = ""

    # Iterate over each detection
    for det in od_results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        od_label = int(det.cls)
        confidence = det.conf.item()
        stage_label = f"stage{od_label+1}"
        od_score = stage_scores.get(stage_label, 0) * confidence

        if od_score > highest_score:
            highest_score = od_score
            highest_label = stage_label

    # Run image classification
    ic_results = ic_model(image)
    ic_probs = ic_results[0].probs
    ic_label = ic_probs.top1
    ic_confidence = ic_probs.top1conf.item()  # Extract confidence for the top class
    ic_score = stage_scores.get(f"stage{ic_label + 1}", 0)

    # Calculate the final score
    final_score = (highest_score + ic_score) / 2

    return highest_label, highest_score, f"stage{ic_label + 1}", ic_score, final_score, ic_confidence

def open_image():
    """Open a single image and load the folder's images."""
    global current_folder, image_files, current_index

    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        current_folder = os.path.dirname(image_path)
        image_files = [f for f in os.listdir(current_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        image_files.sort()  # Sort files alphabetically
        current_index = image_files.index(os.path.basename(image_path))
        load_image()

def load_image():
    """Load and process the current image."""
    global current_index

    if 0 <= current_index < len(image_files):
        image_path = os.path.join(current_folder, image_files[current_index])
        display_image(image_path)
        highest_label, highest_score, ic_label, ic_score, final_score, ic_confidence = process_image(image_path)
        results_text.delete("1.0", tk.END)
        result = f"Image: {os.path.basename(image_path)}, OD Label: {highest_label}, OD Score: {highest_score:.2f}, IC Label: {ic_label}, IC Score: {ic_score}, IC Confidence: {ic_confidence:.2f}, Final Score: {final_score:.2f}\n"
        results_text.insert(tk.END, result)
        display_score_indicator(final_score)
        display_status(final_score)

def next_image():
    """Load the next image in the folder."""
    global current_index
    if current_index < len(image_files) - 1:
        current_index += 1
        load_image()

def previous_image():
    """Load the previous image in the folder."""
    global current_index
    if current_index > 0:
        current_index -= 1
        load_image()

def display_image(image_path):
    """Display the selected image."""
    img = Image.open(image_path)
    img = img.resize((400, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

def display_score_indicator(final_score):
    """Display a colored circle based on the final score."""
    canvas.delete("all")
    if final_score >= 2.7:
        color = "red"
    elif 2 < final_score < 2.7:
        color = "yellow"
    else:
        color = "green"
    canvas.create_oval(10, 10, 60, 60, fill=color)

def display_status(final_score):
    """Display a status message with appropriate color and size."""
    status_text.delete("1.0", tk.END)
    if final_score >= 2.7:
        message = "\u5efa\u8b70\u56de\u8a3a\u89c0\u5bdf"
        color = "red"
    elif 2 < final_score < 2.7:
        message = "\u9700\u8981\u6ce8\u610f"
        color = "yellow"
    else:
        message = "\u7167\u8b77\u826f\u597d"
        color = "green"
    status_text.tag_configure("status", font=("Helvetica", 16, "bold"), foreground=color)
    status_text.insert("1.0", message, "status")

# Create the main window
root = tk.Tk()
root.title("Image Scoring Application")

# Left frame for image selection and display
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

button = tk.Button(left_frame, text="Open Image", command=open_image)
button.pack(pady=5)

image_label = tk.Label(left_frame)
image_label.pack(pady=10)

navigation_frame = tk.Frame(left_frame)
navigation_frame.pack(pady=10)

prev_button = tk.Button(navigation_frame, text="Previous", command=previous_image)
prev_button.pack(side=tk.LEFT, padx=5)

next_button = tk.Button(navigation_frame, text="Next", command=next_image)
next_button.pack(side=tk.RIGHT, padx=5)

# Right frame for displaying results
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

results_text = tk.Text(right_frame, width=50, height=10)
results_text.pack()

status_text = tk.Text(right_frame, width=50, height=2)
status_text.pack(pady=5)

canvas = tk.Canvas(right_frame, width=70, height=70)
canvas.pack(pady=10)

root.mainloop()
