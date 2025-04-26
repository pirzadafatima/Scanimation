import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, Frame, Button, Label, font


# Step 1: Image Enhancement Functions
def log_transformation(image):
    epsilon = 1e-8                       # Small constant to avoid division by zero
    c = 255 / (np.log(1 + np.max(image) + epsilon))
    transformed = c * np.log(1 + image + epsilon)
    return np.clip(transformed, 0, 255).astype(np.uint8)


def image_negatives_transformation(image):
    transformed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            transformed_image[i, j] = 255 - pixel_value
    return transformed_image


def power_law_transformation(image, gamma=2.2):
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return np.clip(gamma_corrected, 0, 255).astype(np.uint8)


def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = 255 * ((image - min_val) / (max_val - min_val))
    return stretched.astype(np.uint8)


# Step 2: Image Filtration
def process_selected_image(root, selected_image):
    cv2.destroyAllWindows()
    # Noise removal using filtering (Gaussian blur as an example)
    filtered_image = cv2.GaussianBlur(selected_image, (5, 5), 0)
    show_main_gui(root, filtered_image)


# Step 3: Custom colormaps
def create_jet_colormap(image):
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):    # this is in BGR format
        if i < 64:
            # Dark Blue to Light Blue
            colormap[i, 0, :] = [255, int(i * 255 / 64), 0]  # Dark Blue + green = light blue
        elif i < 128:
            # Light Blue to Green
            colormap[i, 0, :] = [255, int(255 - (i - 64) * 255 / 64), 0]  # light blue - blue = green
        elif i < 192:
            # Green to Yellow
            colormap[i, 0, :] = [0, 255, int((i - 128) * 255 / 64)]   # green + red = yellow
        else:
            # Yellow to Red
            colormap[i, 0, :] = [0, (int(255 - (i - 192) * 255 / 64)), 255]  # red = yellow - green
    colored_image = apply_custom_colormap(image, colormap)              # Apply colormap to the grayscale image
    return colored_image


def create_hot_colormap(image):
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):  # this is in BGR format
        if i < 64:
            # Black to Maroon
            colormap[i, 0, :] = [0, 0, int(i * 255 / 64)]  # Black + red = maroon
        elif i < 128:
            # Maroon to Orange
            colormap[i, 0, :] = [0, int((i - 64) * 255 / 64), 255]  # green + red = orange
        elif i < 192:
            # Orange to Yellow
            colormap[i, 0, :] = [0, int((i - 128) * 255 / 64), 255]  # orange + green = yellow
        else:
            # Yellow to White
            colormap[i, 0, :] = [int((i - 192) * 255 / 64), 255, 255]  # yellow + blue = white"""
    colored_image = apply_custom_colormap(image, colormap)  # Apply colormap to the grayscale image
    return colored_image


def create_cool_colormap(image):
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):  # this is in BGR format
        if i < 128:
            # Light blue to darkish blue
            colormap[i, 0, :] = [255, int(255 - i * 255 / 128), 0]  # green + blue = light blue - GREEN = darkish blue
        else:
            # Darkish blue to light purple
            colormap[i, 0, :] = [255, 0, int((i - 128) * 255 / 128)]  # darkish blue + red = purple

    colored_image = apply_custom_colormap(image, colormap)  # Apply colormap to the grayscale image
    return colored_image


def create_copper_colormap(image):
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):  # this is in BGR format
        if i < 85:
            # Black to Dark Brown
            colormap[i, 0, :] = [0, int(i * 66 / 85), int(i * 126 / 85)]  # 0, 66, 126
        elif i < 170:
            # Dark Brown to Light Brown
            blue = min(int((i - 85) * 75 / 85), 75)
            green = min(66 + int((i - 85) * (141 - 66) / 85), 141)
            red = min(126 + int((i - 85) * (214 - 126) / 85), 214)
            colormap[i, 0, :] = [blue, green, red]    # 75,141,214
        else:
            # Light Brown to Skin
            blue = min(75 + int((i - 170) * (126 - 75) / 85), 126)
            green = min(141 + int((i - 170) * (195 - 141) / 85), 195)
            red = min(214 + int((i - 170) * (255 - 214) / 85), 255)
            colormap[i, 0, :] = [blue, green, red]  # 126, 195, 255
    colored_image = apply_custom_colormap(image, colormap)  # Apply colormap to the grayscale image
    return colored_image


def create_magma_colormap(image):
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):    # this is in BGR format
        if i < 64:
            # Black to Purple
            colormap[i, 0, :] = [int(i * 130 / 64), 0, int(i * 65 / 64)]     # 130, 0, 65
        elif i < 128:
            # Purple to pink
            blue = min(130 + int((i - 64) * (143 - 130) / 64), 143)
            green = min(int((i - 64) * 11 / 64), 11)
            red = min(65 + int((i - 64) * (187 - 65) / 64), 187)
            colormap[i, 0, :] = [blue, green, red]                  # 143, 11, 187
        elif i < 192:
            # Pink to orange
            blue = max(143 - int((i - 128) * (143 - 97) / 64), 97)
            green = min(11 + int((i - 128) * (155 - 11) / 64), 155)
            red = min(187 + int((i - 128) * (243 - 187) / 64), 243)
            colormap[i, 0, :] = [blue, green, red]                   # 97, 155, 243
        else:
            # Orange to yellow
            blue = min(97 + int((i - 192) * (157 - 97) / 64), 157)
            green = min(155 + int((i - 192) * (237 - 155) / 64), 237)
            red = min(243 + int((i - 192) * (249 - 243) / 64), 249)
            colormap[i, 0, :] = [blue, green, red]                    # 157, 237, 249
    colored_image = apply_custom_colormap(image, colormap)              # Apply colormap to the grayscale image
    return colored_image


def apply_custom_colormap(image, colormap):
    # Ensure that the colormap has shape (256, 1, 3)
    if colormap.shape != (256, 1, 3):
        raise ValueError("Colormap must have shape (256, 1, 3)")

    # Apply colormap to the grayscale image
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity = image[i, j]
            colored_image[i, j, :] = colormap[intensity]

    return colored_image


global selected_image


def enhance_image(image_path):   # step 2

    root = tk.Tk()
    root.title("Enhanced Image Selection")
    root.geometry("1520x900")
    palette = {
        "Tradewind": "#5faba8",
        "Neptune": "#8bc1b8",
        "Opal": "#a3c8c8",
        "Tiara": "#c6d2d2",
        "Cararra": "#e8e9e2"
    }

    title_font = font.Font(size=10, weight="bold")

    # Apply custom color scheme to root window
    root.config(bg=palette["Tradewind"])

    # Frame for color mapping options
    colormap_frame = Frame(root, bg=palette["Tradewind"])
    colormap_frame.pack(expand=True, fill='both', padx=10, pady=10)  # Expand to fill space

    option_frame = Frame(colormap_frame, bg=palette["Tradewind"])
    option_frame.pack(side="top", fill="x", padx=10, pady=10)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply transformations
    log_transformed = log_transformation(image)
    negatives_transformed = image_negatives_transformation(image)
    power_law_trans = power_law_transformation(log_transformed)
    power_law_trans1 = power_law_transformation(image)
    contrast_stretched = contrast_stretching(image)

    # Display the enhanced images
    cv2.imshow("Original Image", image)
    cv2.imshow("Image Negatives Transformation", negatives_transformed)
    cv2.imshow("Contrast Stretching", contrast_stretched)
    cv2.imshow("Log Transformation ", log_transformed)
    cv2.imshow("Power Law Transformation on Log Transformed", power_law_trans)
    cv2.imshow("Power Law Transformation on original", power_law_trans1)

    def on_selection(selection):
        global selected_image
        if selection == "Original Image":
            selected_image = image
            print("Original Image selected.")
        if selection == "Power Law Transformation on original":
            selected_image = power_law_trans1
            print("Power Law Transformation on original")
        if selection == "Log Transformation":
            selected_image = log_transformed
            print("Log-transformed image selected.")
        if selection == "Image Negatives Transformation":
            selected_image = negatives_transformed
            print("Image Negatives Transformation selected.")
        if selection == "Contrast Stretching":
            selected_image = contrast_stretched
            print("Contrast Stretching selected.")
        if selection == "Power Law Transformation on Log Transformed":
            selected_image = power_law_trans
            print("Power Law Transformation on Log Transformed selected.")

        # Perform further processing with the selected image (going to step 3)
        process_selected_image(root, selected_image)

    # Create buttons for each enhanced image
    original_button = Button(option_frame, text="Original Image", command=lambda: on_selection("Original Image"),
                             font=title_font)
    original_button.config(bg=palette["Neptune"], fg="white", width=30, height=2, bd=0, padx=10, pady=5,
                           activebackground=palette["Opal"], activeforeground="white")
    original_button.pack(pady=10)

    po_button = Button(option_frame, text="Power Law Transformation on original",
                       command=lambda: on_selection("Power Law Transformation on original"), font=title_font)
    po_button.config(bg=palette["Neptune"], fg="white", width=30, height=2, bd=0, padx=10, pady=5,
                     activebackground=palette["Opal"], activeforeground="white")
    po_button.pack(pady=10)

    log_button = Button(option_frame, text="Log Transformation", command=lambda: on_selection("Log Transformation"),
                        font=title_font)
    log_button.config(bg=palette["Neptune"], fg="white", width=30, height=2, bd=0, padx=10, pady=5,
                      activebackground=palette["Opal"], activeforeground="white")
    log_button.pack(pady=10)

    negative_button = Button(option_frame, text="Image Negatives Transformation",
                             command=lambda: on_selection("Image Negatives Transformation"), font=title_font)
    negative_button.config(bg=palette["Neptune"], fg="white", width=30, height=2, bd=0, padx=10, pady=5,
                           activebackground=palette["Opal"], activeforeground="white")
    negative_button.pack(pady=10)

    cs_button = Button(option_frame, text="Contrast Stretching", command=lambda: on_selection("Contrast Stretching"),
                       font=title_font)
    cs_button.config(bg=palette["Neptune"], fg="white", width=30, height=2, bd=0, padx=10, pady=5,
                     activebackground=palette["Opal"], activeforeground="white")
    cs_button.pack(pady=10)

    pl_button = Button(option_frame, text="Power Law Transformation on Log Transformed",
                       command=lambda: on_selection("Power Law Transformation on Log Transformed"), font=title_font)
    pl_button.config(bg=palette["Neptune"], fg="white", width=40, height=2, bd=0, padx=10, pady=5,
                     activebackground=palette["Opal"], activeforeground="white")
    pl_button.pack(pady=10)

    root.mainloop()


def apply_map(filtered_image, colormap_func):
    # Apply custom colormap
    colored_image = colormap_func(filtered_image)

    # Display the enhanced image
    cv2.imshow("Filtered Image", filtered_image)
    cv2.imshow("Colormap Image", colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_image(root):

    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename()
    show_enhance_gui(file_path, root)
    """if file_path:
        enhance_image(file_path)
    else:
        print("No file selected.")"""


def select_image_gui():
    root = tk.Tk()

    root.title("Scanimation Image Selection")
    root.geometry("1520x900")
    palette = {
        "Tradewind": "#5faba8",
        "Neptune": "#8bc1b8",
        "Opal": "#a3c8c8",
        "Tiara": "#c6d2d2",
        "Cararra": "#e8e9e2"
    }

    title_font = font.Font(size=20, weight="bold")
    title_color = "#e8e9e2"  # White color

    # Apply custom color scheme to root window
    root.config(bg=palette["Tradewind"])

    # Create title label
    title_label = tk.Label(root, text="Ready to add vibrant hues to your medical images? Select an image and let's "
                                      "bring your diagnosis to life with pseudocoloring!", bg=palette["Tradewind"],
                           fg=title_color, font=("Georgia", 15, "bold"))
    title_label.pack(pady=(100, 0))

    # Create frame for button
    button_frame = tk.Frame(root, bg=palette["Tradewind"])
    button_frame.pack(pady=(150, 0))

    # Create select image button
    button = tk.Button(button_frame, text="Select Image", command=lambda: select_image(root), font=title_font)
    button.config(bg=palette["Neptune"], fg="white", width=15, height=2, bd=0, padx=10, pady=5,
                  activebackground=palette["Opal"], activeforeground="white")
    button.pack()

    root.mainloop()


# Create the GUI
def create_gui(filtered_image):
    root = tk.Tk()
    root.title("Scanimation Colormap Enhancement")
    root.geometry("1520x900")

    # Define custom color scheme
    palette = {
        "Tradewind": "#5faba8",
        "Neptune": "#8bc1b8",
        "Opal": "#a3c8c8",
        "Tiara": "#c6d2d2",
        "Cararra": "#e8e9e2"
    }

    title_font = font.Font(size=20, weight="bold")
    title_color = "#e8e9e2"  # White color

    # Apply custom color scheme to root window
    root.config(bg=palette["Tradewind"])

    # Title Label
    title_label = Label(root, text="Scanimation", bg=palette["Tradewind"], fg=title_color, font=("Georgia", 40, "bold"))
    title_label.pack(pady=3)

    # Frame for color mapping options
    colormap_frame = Frame(root, bg=palette["Tradewind"])
    colormap_frame.pack(expand=True, fill='both', padx=10, pady=10)  # Expand to fill space

    # Color mapping options
    colormap_buttons = [
        ("Jet", create_jet_colormap, "jet.png"),
        ("Hot", create_hot_colormap, "hot.png"),
        ("Cool", create_cool_colormap, "cool.png"),
        ("Copper", create_copper_colormap, "copper.png"),        ("Magma", create_magma_colormap, "magma.png")
    ]

    for name, func, thumbnail_path in colormap_buttons:
        # Frame for colormap option
        option_frame = Frame(colormap_frame, bg=palette["Tradewind"])
        option_frame.pack(side="top", fill="x", padx=10, pady=10)

        # Create button with title
        button = Button(option_frame, text=name, command=lambda func=func: apply_map(filtered_image, func),
                        font=title_font)
        button.config(bg=palette["Neptune"], fg="white", width=10, height=2, bd=0, padx=10, pady=5,
                      activebackground=palette["Opal"], activeforeground="white")
        button.pack(side="left", padx=10)  # Add padding between button and image

        # Load thumbnail image
        thumbnail = Image.open(thumbnail_path)
        thumbnail = thumbnail.resize((300, 50))  # Resize thumbnail image
        thumbnail = ImageTk.PhotoImage(thumbnail)

        # Create Label to display image
        image_label = Label(option_frame, image=thumbnail, bg=palette["Tradewind"])
        image_label.image = thumbnail
        image_label.pack(side="left")  # Add padding between images

    root.mainloop()


def start_page(root):
    # Define font

    start_font = font.Font(size=16, weight="bold")

    # Starting Page Frame
    start_frame = Frame(root, bg="#5faba8")  # White background
    start_frame.pack(fill="both", expand=True)

    # Logo
    logo_image = tk.PhotoImage(file="Scanimation.png")

    # Scale down the image by a factor of 2 (halve the size)
    logo_image = logo_image.subsample(2, 2)
    logo_label = Label(start_frame, image=logo_image, bg="#5faba8")
    logo_label.image = logo_image
    logo_label.place(relx=0.5, rely=0.2, anchor="center")

    # Title Label
    title_label = Label(start_frame, text="Scanimation", font=("Georgia", 40, "bold"),
                        bg="#5faba8", fg="#e8e9e2")
    title_label.place(relx=0.5, rely=0.4, anchor="center")

    # Title Label
    title_label = Label(start_frame, text="Vivid Vision: Unveiling Health in HD", font=("Georgia", 20, "bold"),
                        bg="#5faba8", fg="#e8e9e2")
    title_label.place(relx=0.5, rely=0.6, anchor="center")

    # Start Button
    start_button = Button(start_frame, text="Start", command=lambda: show_select_image_page(root), font=start_font,
                          bg="#5faba8", fg="#e8e9e2", width=10, height=2)
    start_button.place(relx=0.5, rely=0.8, anchor="center")  #


def show_enhance_gui(file_path, root):
    # Destroy the starting page
    root.destroy()

    # Create the main GUI
    enhance_image(file_path)


def show_main_gui(root, filtered_image):
    # Destroy the starting page
    root.destroy()

    # Create the main GUI
    create_gui(filtered_image)


def show_select_image_page(root):
    # Destroy the starting page
    root.destroy()

    # Change this to select image page
    select_image_gui()


def main():
    root = tk.Tk()
    root.title("Welcome to Scanimation")
    root.geometry("1520x900")

    # Show the starting page
    start_page(root)

    root.mainloop()

if __name__ == "__main__":
    main()