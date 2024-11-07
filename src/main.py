import cv2
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import filedialog
from torchvision.transforms import v2
import torch.nn as nn
import os


app = ttk.Window(themename="darkly")
app.title("UNO Card Detector")
app.geometry("1200x800")

# Set constants for image processing
IMG_WIDTH = 384
IMG_HEIGHT = 216
IMG_CHANNELS = 3

device = torch.device("cpu")


class UnoSymbolClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # First convolutional block
        self.conv1_block = nn.Sequential(
            # 1x1 convolution to reduce channel dimensions
            nn.Conv2d(3, 8, 1, 1),
            # Batch normalization for stability
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Padding to maintain spatial dimensions
            nn.ReflectionPad2d(1),
            # 3x3 convolution to capture more spatial features
            nn.Conv2d(8, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Max pooling to reduce spatial dimensions
            nn.MaxPool2d(2, 2),
        )

        # Second convolutional block
        self.conv2_block = nn.Sequential(
            nn.Conv2d(16, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Third convolutional block
        self.conv3_block = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fourth convolutional block
        self.conv4_block = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Flatten layer to convert 2D feature maps to 1D feature vectors
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Dropout for regularization
            nn.Dropout(0.4)
        )

        # Second fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Third fully connected layer
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Output layer
        self.fc4 = nn.Linear(64, 54)  # 54 classes for classification

    def forward(self, x) -> torch.utils.data.Dataset:
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = self.conv4_block(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


# Load the pre-trained model
model = UnoSymbolClassifier()
state_dict = torch.load('src/data/full_symbol_classifier.pth', map_location=device)
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()


image_transform = v2.Compose([
    v2.Resize((IMG_WIDTH, IMG_HEIGHT)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5479778051376343, 0.526210367679596, 0.4944702088832855], std=[
                 0.2237844169139862, 0.23763211071491241, 0.26044926047325134])
])


# UNO card class names
class_names = ['0_blue', '0_green', '0_red', '0_yellow', '1_blue', '1_green', '1_red', '1_yellow', '2_blue', '2_green', '2_red', '2_yellow',
               '3_blue', '3_green', '3_red', '3_yellow', '4_blue', '4_green', '4_red', '4_yellow', '5_blue', '5_green', '5_red', '5_yellow',
               '6_blue', '6_green', '6_red', '6_yellow', '7_blue', '7_green', '7_red', '7_yellow', '8_blue', '8_green', '8_red', '8_yellow',
               '9_blue', '9_green', '9_red', '9_yellow', 'wild_color', 'draw2_blue', 'draw2_green', 'draw2_red', 'draw2_yellow', 'wild_draw',
               'reverse_blue', 'reverse_green', 'reverse_red', 'reverse_yellow', 'skip_blue', 'skip_green', 'skip_red', 'skip_yellow']


def start_screen():
    # Clear any previous widgets
    for widget in app.winfo_children():
        widget.destroy()

    title_frame = ttk.Frame(app)
    title_frame.pack(pady=50)

    # Load images for the icons
    img_before = Image.open("src/data/static/logo/unologo.png")
    img_before.thumbnail((100, 100), Image.LANCZOS)
    img_before = ImageTk.PhotoImage(img_before)

    img_after = Image.open("src/data/static/logo/unologo1.png")
    img_after.thumbnail((100, 100), Image.LANCZOS)
    img_after = ImageTk.PhotoImage(img_after)

    label = ttk.Label(title_frame, text="UNO Card Detector",
                      font=("Arial", 48, "bold"))

    # Image labels to display images
    img_before_label = ttk.Label(title_frame, image=img_before)
    img_after_label = ttk.Label(title_frame, image=img_after)

    # Keep a reference to the images to prevent garbage collection
    img_before_label.image = img_before
    img_after_label.image = img_after

    img_before_label.pack(side="left", padx=10)
    label.pack(side="left", padx=10)
    img_after_label.pack(side="left", padx=10)

    # Create a custom style for the buttons
    style = ttk.Style()
    style.configure('Custom.TButton', font=('Arial', 16), background='red')

    button_upload = ttk.Button(
        app, text="Upload Image", style='Custom.TButton', bootstyle="danger", command=open_image)
    button_upload.pack(pady=(100, 0), ipadx=50, ipady=15)

    button_live = ttk.Button(app, text="Use Live Camera", style='Custom.TButton',
                             bootstyle="danger", command=live_camera_screen)
    button_live.pack(pady=30, ipadx=30, ipady=15)


def detect_card(image):
    transformed_image = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(transformed_image)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]


def live_camera_screen():
    for widget in app.winfo_children():
        widget.destroy()

    # cap = cv2.VideoCapture('http://x.x.x.x:8080/video') # Use IP Webcam for Android
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    left_frame = ttk.Frame(app, width=850, height=800, bootstyle="dark")
    left_frame.pack(side=LEFT, fill=BOTH, expand=True)

    video_display = ttk.Label(left_frame)
    video_display.place(relx=0.5, rely=0.5, anchor="center")

    right_frame = ttk.Frame(app, width=350, height=800, bootstyle="secondary")
    right_frame.pack(side=RIGHT, fill=Y)

    title_label = tk.Label(right_frame, text="Detected Card:", font=(
        "Arial", 16), fg="white", width=30)
    title_label.pack(pady=(200, 0))
    result_label = tk.Label(right_frame, text="None",
                            font=("Arial", 16), fg="white", width=30)
    result_label.pack(pady=0)

    card_image_label = ttk.Label(right_frame)
    card_image_label.pack(pady=30)

    back_button = ttk.Button(right_frame, text="Back to Home",
                             bootstyle="danger-outline", command=lambda: [cap.release(), start_screen()])
    back_button.pack(pady=60, ipadx=20, ipady=10)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Define the bounding box dimensions
            frame_height, frame_width = frame.shape[:2]
            box_width, box_height = 300, 400
            top_left_x = frame_width // 2 - box_width // 2
            top_left_y = frame_height // 2 - box_height // 2
            bottom_right_x = top_left_x + box_width
            bottom_right_y = top_left_y + box_height

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (top_left_x, top_left_y),
                          (bottom_right_x, bottom_right_y), (255, 0, 0), 2)

            # Crop the frame to the bounding box
            cropped_frame = frame[top_left_y:bottom_right_y,
                                  top_left_x:bottom_right_x]

            # Convert the cropped frame to PIL format for processing
            cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_rgb)

            # Detect UNO card in the cropped frame
            predicted_class = detect_card(pil_image)
            result_label.config(text=f"{predicted_class.replace('_', ' ')}")

            # Display the detected card image if available
            card_image_path = os.path.join(
                "src/data/static/cards", f"{predicted_class}.jpg")
            if os.path.exists(card_image_path):
                card_image = Image.open(card_image_path)

                # Resize the image proportionally to fit in a smaller display area
                max_size = (200, 200)
                card_image.thumbnail(max_size, Image.LANCZOS)

                # Update the card image display
                card_image_tk = ImageTk.PhotoImage(card_image)

                card_image_label.imgtk = card_image_tk
                card_image_label.config(image=card_image_tk)
            else:
                # Clear the label if the image is not found
                card_image_label.config(image="")

            # Convert frame with bounding box to display in the Tkinter window
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=frame_pil)
            video_display.imgtk = imgtk
            video_display.configure(image=imgtk)

        # Schedule the next frame update
        app.after(10, update_frame)

    update_frame()

    # Ensure the camera is released when navigating back
    app.protocol("WM_DELETE_WINDOW", lambda: [cap.release(), app.destroy()])

# UI for Upload Image


def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Load the image in full size and convert it to RGB
        pil_image = Image.open(file_path).convert("RGB")

        max_size = (500, 500)
        pil_image.thumbnail(max_size, Image.LANCZOS)

        predicted_class = detect_card(pil_image)

        for widget in app.winfo_children():
            widget.destroy()

        left_frame = ttk.Frame(app, width=850, height=800, bootstyle="dark")
        left_frame.pack(side=LEFT, fill=BOTH, expand=True)

        img_display = ttk.Label(left_frame)
        img = ImageTk.PhotoImage(image=pil_image)
        img_display.imgtk = img
        img_display.configure(image=img)
        img_display.place(relx=0.5, rely=0.5, anchor="center")

        right_frame = ttk.Frame(
            app, width=350, height=800, bootstyle="secondary")
        right_frame.pack(side=RIGHT, fill=Y)

        title_label = tk.Label(right_frame, text="Detected Card :", font=(
            "Arial", 16), fg="white", width=30)
        title_label.pack(pady=(200, 0))
        result_label = tk.Label(right_frame, text=f"{predicted_class.replace('_', ' ')}", font=(
            "Arial", 16), fg="white", width=30)
        result_label.pack(pady=0)

        card_image_label = ttk.Label(right_frame)
        card_image_label.pack(pady=30)

        # Display the detected card image if available
        card_image_path = os.path.join(
            "src/data/static/cards", f"{predicted_class}.jpg")
        if os.path.exists(card_image_path):
            card_image = Image.open(card_image_path)

            max_size = (200, 200)
            card_image.thumbnail(max_size, Image.LANCZOS)

            card_image_tk = ImageTk.PhotoImage(card_image)

            card_image_label.imgtk = card_image_tk
            card_image_label.config(image=card_image_tk)
        else:
            # Clear the label if the image is not found
            card_image_label.config(image="")

        back_button = ttk.Button(
            right_frame, text="Back to Home", bootstyle="danger-outline", command=start_screen)
        back_button.pack(pady=60, ipadx=20, ipady=10)


start_screen()
app.mainloop()
