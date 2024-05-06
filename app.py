import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


class TextToImageApp(tk.Tk):
    def __init__(self, model_id, device, guidance_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initializing the base Tk Class
        self.geometry("600x700")  # Setting the Dimensions of the window
        self.title("Text to Image Generator")  # Setting the title of the APP

        # Setting the appearance as the system
        ctk.set_appearance_mode("light")

        # Sets the color of the widgets to green
        # ctk.set_default_color_theme("green")

        # Creating the stable diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=auth_token
        )
        self.pipe.to(device)

        self.device = device
        self.guidance_scale = guidance_scale

        # Initialize UI components
        self.create_widgets()

    def create_widgets(self):
        # Create the GUI components
        frame = ctk.CTkFrame(self)
        frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Prompt Entry
        self.prompt_entry = ctk.CTkEntry(
            frame, placeholder_text="Enter text for the image....", height=40, width=450
        )
        self.prompt_entry.grid(row=0, column=0, columnspan=2, pady=10)

        # Creating the Genereate Button
        self.generate_button = ctk.CTkButton(
            frame,
            height=40,
            width=120,
            font=("Arial", 16),
            text_color="white",
            fg_color="blue",
            text="Generate",
            command=self.generate_image
        )
        self.generate_button.grid(row=1, column=0, padx=10, pady=10)

        # Button to clear the prompt
        self.clear_button = ctk.CTkButton(
            frame,
            height=40,
            width=120,
            font=("Arial", 16),
            text="Clear",
            fg_color="red",
            command=self.clear_prompt
        )
        self.clear_button.grid(row=1, column=1, padx=10, pady=10)

        # Getting the image here
        self.image_display = ctk.CTkLabel(frame, height=400, width=500, text="")
        self.image_display.grid(row=2, column=0, columnspan=2, pady=20)

    def generate_image(self):
        prompt_text = self.prompt_entry.get()

        with autocast(self.device):
            generated_image = self.pipe(prompt_text, guidance_scale=self.guidance_scale)["images"][0]

        image_path = f"generatedimage.png"
        generated_image.save(image_path)

        img = ImageTk.PhotoImage(generated_image)
        self.image_display.configure(image=img)
        self.image_display.image = img

    def clear_prompt(self):
        self.prompt_entry.delete(0, tk.END)


# Constants
MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
GUIDANCE_SCALE = 7

# Create and run the application
app = TextToImageApp(model_id=MODEL_ID, device=DEVICE, guidance_scale=GUIDANCE_SCALE)
app.mainloop()
