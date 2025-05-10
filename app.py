import streamlit as st
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from usability_analysis import analyze_currency

STANDARD_NOTES_DIR = 'standard_notes'
MODEL_WEIGHTS_PATH = 'dataset_customcnn_model_trained.pth'
UPLOAD_DIR = 'uploaded_notes'

os.makedirs(UPLOAD_DIR, exist_ok=True)

class_names = ['1', '500', '100', '1000', '50', '20', '200', '5', '2', '10']

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Currency Note Usability Analyzer (Front + Back)")

uploaded_front = st.file_uploader("Upload the FRONT side of the banknote", type=["jpg", "png", "jpeg"])
uploaded_back = st.file_uploader("Upload the BACK side of the banknote", type=["jpg", "png", "jpeg"])

if uploaded_front and uploaded_back:
    front_img = Image.open(uploaded_front).convert('RGB')
    front_image_path = os.path.join(UPLOAD_DIR, f"front_{uploaded_front.name}")
    front_img.save(front_image_path)
    st.image(front_img, caption='Uploaded Front Note', use_container_width=True)

    back_img = Image.open(uploaded_back).convert('RGB')
    back_image_path = os.path.join(UPLOAD_DIR, f"back_{uploaded_back.name}")
    back_img.save(back_image_path)
    st.image(back_img, caption='Uploaded Back Note', use_container_width=True)

    with st.spinner("Classifying the note denomination (using front)..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CustomCNN(num_classes=len(class_names))
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model = model.to(device)
        model.eval()

        input_tensor = transform(front_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(probabilities, 1)

        denomination = class_names[predicted_idx.item()]
        confidence = probabilities[0][predicted_idx].item()

    st.success(f"Predicted Note: {denomination} Taka (Confidence: {confidence:.2f})")

    standard_front_filename = f"{denomination}_standard.jpg"
    standard_back_filename = f"{denomination}_back_standard.jpg"
    standard_front_path = os.path.join(STANDARD_NOTES_DIR, standard_front_filename)
    standard_back_path = os.path.join(STANDARD_NOTES_DIR, standard_back_filename)

    st.header("Front Side Usability Analysis")
    if not os.path.exists(standard_front_path):
        st.error(f"Standard front note not found: {standard_front_filename}")
    else:
        st.image(standard_front_path, caption=f'Standard Note (Front - {denomination})', use_container_width=True)

        with st.spinner("Running usability analysis for front..."):
            try:
                result_front = analyze_currency(front_image_path, standard_front_path)

                st.subheader("Damage Analysis (Front)")
                st.write(f"- Area Damaged: {result_front.get('binary_damage', 0):.1f}%")
                st.write(f"- Color Distortion: {result_front.get('rgb_damage', 0):.1f}%")

                st.subheader("Edge & Corner Damage (Front)")
                edge_report = result_front.get('edge_corner_report', {})
                if edge_report:
                    for region, data in edge_report.items():
                        status = "Damaged" if data['is_damaged'] else "OK"
                        st.write(f"- {region}: {status} ({data['overlap_percentage']:.1f}% damaged)")
                else:
                    st.write("No edge or corner damage data available.")

                st.subheader("Top 5 Damaged Areas (Front)")
                damaged_areas = result_front.get('top_damaged_areas', [])
                if damaged_areas:
                    for idx, area in enumerate(damaged_areas, 1):
                        st.write(f"{idx}. Location: {area['zone']}, Area: {area['area_px']} px")
                else:
                    st.write("No significant damaged areas found.")

            except Exception as e:
                st.error(f"Error during front analysis: {str(e)}")

    st.header("Back Side Usability Analysis")
    if not os.path.exists(standard_back_path):
        st.error(f"Standard back note not found: {standard_back_filename}")
    else:
        st.image(standard_back_path, caption=f'Standard Note (Back - {denomination})', use_container_width=True)

        with st.spinner("Running usability analysis for back..."):
            try:
                result_back = analyze_currency(back_image_path, standard_back_path)

                st.subheader("Damage Analysis (Back)")
                st.write(f"- Area Damaged: {result_back.get('binary_damage', 0):.1f}%")
                st.write(f"- Color Distortion: {result_back.get('rgb_damage', 0):.1f}%")

                st.subheader("Edge & Corner Damage (Back)")
                edge_report = result_back.get('edge_corner_report', {})
                if edge_report:
                    for region, data in edge_report.items():
                        status = "Damaged" if data['is_damaged'] else "OK"
                        st.write(f"- {region}: {status} ({data['overlap_percentage']:.1f}% damaged)")
                else:
                    st.write("No edge or corner damage data available.")

                st.subheader("Top 5 Damaged Areas (Back)")
                damaged_areas = result_back.get('top_damaged_areas', [])
                if damaged_areas:
                    for idx, area in enumerate(damaged_areas, 1):
                        st.write(f"{idx}. Location: {area['zone']}, Area: {area['area_px']} px")
                else:
                    st.write("No significant damaged areas found.")

            except Exception as e:
                st.error(f"Error during back analysis: {str(e)}")
