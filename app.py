import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import cast_model
from cast_model import CASTModel
from torchvision.io import read_video
import torch
import torchvision.transforms as transforms
from torchvision.io import write_video
import torch.nn.functional as F


def resize_frame(frame, size):
            # This function assumes input frame is a tensor
            return F.interpolate(frame.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

transform_video = transforms.Compose([
            # First convert to float and scale to [0, 1]
            transforms.Lambda(lambda x: x.float() / 255.0),
            # Apply resizing within the lambda function using the custom resize_frame function

            transforms.Lambda(lambda x: resize_frame(x, (288, 288))), #was self.opt.loadsize and not 286
            # Normalize the frames
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

transform_image = transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        

def stylize():
    cm = CASTModel()
    # Read video frames
    video_frames, audio, info = read_video(mp4_file, pts_unit='sec')
    # Convert video frames to tensor and normalize
    A = torch.stack([transform_video(frame.permute(2,0,1)) for frame in video_frames])
    B = transform_image(img.convert('RGB'))
    dic = {'A': A, 'B': B}
    cm.set_input(dic)
    output = torch.stack(cm.forward()) # to tensor
    filename = 'out_video.mp4'
    video_tensor = torch.stack([frame.permute(1, 2, 0) for frame in output]).cpu()
    reconstructed_video = write_video(filename, video_tensor, fps=info['video_fps'])
    
    return reconstructed_video


def upload_mp4():
    global mp4_file, video_label, cap
    mp4_file = filedialog.askopenfilename(title="Select an MP4 File", filetypes=[("MP4 files", "*.mp4")])
    if mp4_file:
        print(f"MP4 file selected: {mp4_file}")
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(mp4_file)
        # Start the update process.
        update_frame(cap, video_label)

def upload_image():
    global image_file, image_label
    global img
    image_file = filedialog.askopenfilename(title="Select an Image", filetypes=[("JPEG files", "*.jpeg"), ("JPG files", "*.jpg"), ("PNG files", "*.png")])
    if image_file:
        print(f"Image file selected: {image_file}")
        img = Image.open(image_file)
        img = img.resize((300, 200), Image.ANTIALIAS)  # Resize the image immediately
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)

def update_frame(cap, label, delay=33):
    # Capture the next frame from the video.
    ret, frame = cap.read()

    if ret:
        # Convert the frame to RGB (for Tkinter compatibility) and resize.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (300, 200))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        # Schedule the next update with a specified delay.
        label.after(delay, update_frame, cap, label, delay)
    else:
        # If the video is over, reset to the beginning.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        update_frame(cap, label, delay)

# Create the main window
root = tk.Tk()
root.title("V-CAST: Stylize a Video")
root.configure(bg='#222222')  # Dark grey background

# Keep a reference to the VideoCapture object.
cap = None

# Frame for buttons
button_frame = tk.Frame(root, bg='#2D2D2D')
button_frame.pack(fill=tk.X, pady=10)

# Frame for displaying the image and video
display_frame = tk.Frame(root, bg='#2D2D2D')
display_frame.pack(pady=10)

# Create labels for displaying the image and video within the display frame
image_label = tk.Label(display_frame, width=400, height=300)
image_label.grid(row=0, column=0, padx=10, pady=10)

video_label = tk.Label(display_frame, width=400, height=300)
video_label.grid(row=0, column=1, padx=10, pady=10)

# Stylish buttons
button_style = {
    'bg': '#393939',  # Dark grayish color for button background
    'fg': '#333333',  # White color text for contrast
    'activebackground': '#4F4F4F',  # Slightly lighter gray when the button is clicked
    'activeforeground': '#333333',
    'borderwidth': '1',
    'font': ('Helvetica', 12)
}
mp4_button = tk.Button(button_frame, text="Upload Video", command=upload_mp4, **button_style)
mp4_button.pack(side=tk.LEFT, padx=20, pady=10)

image_button = tk.Button(button_frame, text="Upload Style Image", command=upload_image, **button_style)
image_button.pack(side=tk.LEFT, padx=20, pady=10)

# Button for stylizing the image or video, which currently does nothing
stylize_button = tk.Button(button_frame, text="Stylize", command=stylize, **button_style)
stylize_button.pack(side=tk.LEFT, padx=(250, 20))  # Adjust the padding to place it to the right

# Start the event loop
root.mainloop()
