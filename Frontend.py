import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import pickle
import process_image as pi  # Assuming process_image.py contains your image processing logic
from MultiInputModel import MultiInputModel
# Define the class names for predictions
class_names = {
    0: 'Dermatofibroma',
    1: 'Melanocytic nevi',
    2: 'Benign keratosis-like lesion',
    3: 'Melanoma',
    4: 'Vascular lesion',
    5: 'Basal cell carcinoma',
    6: 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease'
}

# Load the model
# Load the model
def load_model(model_path):
    model = MultiInputModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

# Load the model
model_path = "Best_multi_input_model.pth"  # Update with your model path
model = load_model(model_path)

def predict(image, age_input, gender_input, localization_input, model):
    # Preprocess the test image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)

    # Preprocess the demographic inputs
    age_input = float(age_input)
    gender_input = 1 if gender_input.lower() == 'female' else 0
    localization_map = {'head/neck': 0, 'lower extremity': 1, 'oral/genital': 2, 'palms/soles': 3, 'torso': 4, 'upper extremity': 5}
    localization_input = localization_map[localization_input.lower()]
    demographic_inputs = torch.tensor([age_input, localization_input, gender_input], dtype=torch.float32).unsqueeze(0)

    # Use the model to predict
    model.eval()
    with torch.no_grad():
        # Forward pass
        output = model(input_image, demographic_inputs)

        predicted_class = torch.argmax(output, dim=1).item()

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name

def main():
    st.title("Skin Lesion Classifier")
    st.write("Upload an image and provide age, gender, and lesion location details")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Convert the uploaded image to a PIL Image object
        pil_image = Image.open(uploaded_image)

        # Process the image
        processed_image = pi.process_single_image(np.array(pil_image))

        # Display the processed image
        st.image(processed_image, channels="RGB", caption="Processed Image")

        age = st.number_input("Enter Age", min_value=0, max_value=150, value=20, step=1)
        sex = st.radio("Select Gender", options=["Male", "Female"])
        dropdown_options = ['head/neck','lower extremity', 'oral/genital', 'palms/soles', 'torso', 'upper extremity']
        selected_option = st.selectbox("Select Lesion Location", options=dropdown_options)

        # Add a button to trigger prediction
        if st.button('Predict'):
            # Perform prediction here
            predicted_class_name = predict(pil_image, age, sex, selected_option, model)
            st.write("Predicted Class:", predicted_class_name)

if __name__ == "__main__":
    main()