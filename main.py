from tkinter import Tk, Button, Label
from PIL import Image, ImageTk  # Use Pillow for advanced image handling
import subprocess  # To run the external Python file

def open_next_page():
    # Run the external Python file
    subprocess.run(["python", "face_recognition.py"])

# Create the main window
root = Tk()
root.title("Face Recognition Project")
root.geometry("800x600")  # Window size

# Load and set the background image
image = Image.open("bg2.jpg")  # Replace with your local image path
bg_image = ImageTk.PhotoImage(image)
bg_label = Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)  # Cover the full window

# Create a rounded button
button = Button(
    root, 
    text="Scan Face", 
    font=("Arial", 14), 
    bg="white",  # Button background color
    fg="navy blue",   # Button text color
    command=open_next_page, 
    relief="flat"
)
button.place(relx=0.5, rely=0.7, anchor="center", width=200, height=50)  # Place in the center

# Add styling for rounded corners using canvas
button.configure(highlightthickness=0, bd=0)

root.mainloop()