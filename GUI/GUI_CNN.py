import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('../Models/CNN_MODEL.h5')


# Function to perform face recognition
def recognize_face(image_path):
    # Load the image directly from the .npy file
    img_array = np.load(image_path)

    # If the image is flattened, reshape it to its original shape (e.g., (64, 64) for grayscale)
    original_shape = (64, 64)
    img_array = img_array.reshape(original_shape)

    # Expand dimensions to match the model input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Add a channel dimension (if needed)
    # img_array = np.expand_dims(img_array, axis=-1)  # Uncomment if your model expects a single channel

    # Normalize or preprocess the image as needed
    # img_array = preprocess_input(img_array)  # Uncomment if preprocessing is required

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted label
    predicted_label = np.argmax(predictions)

    return predicted_label


# Function to open a file dialog and recognize faces from an image
def recognize_faces_from_file():
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("NumPy files", "*.npy")])

    if file_path:
        predicted_label = recognize_face(file_path)
        result_label.config(text=f"Predicted Person: {predicted_label}")


# GUI setup
root = tk.Tk()
root.title("Face Recognition")

# Create and set up the main frame
root.geometry("700x320+200+80")
# root.eval('tk::PlaceWindow . center')
root['background'] = '#8B0A50'
# main_frame = tk.Frame(root)
# main_frame.pack(padx=250, pady=150)

result_label = tk.Label(root, bg='deeppink4', text="Predicted Person: ", font=('Comic Sans MS', 30, 'bold'))
result_label.pack(pady=50)
# Create a button to open a file dialog
open_button = tk.Button(root, text="Upload Image", borderwidth=5, font=('Comic Sans MS', 10, 'bold'), fg='black',
                        bg='moccasin', height=2, width=20, command=recognize_faces_from_file)
open_button.pack(pady=3)

# Create a label to display the recognition result


# Run the GUI
root.mainloop()