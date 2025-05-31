import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from pymongo import MongoClient
from bson import ObjectId
import matplotlib.pyplot as plt
import random


st.set_page_config(page_title="Plant Disease Full Analysis", layout="wide")

st.title("🌿 Plant Disease Detector")
st.markdown("""
Welcome to the Plant Disease Detector application. This tool allows you to:
- Understand dataset characteristics.
- Predict diseases from leaf images using a trained CNN model.
- Receive actionable treatment advice (Prescriptive Analysis)
""")

@st.cache_resource
def get_mongo_client():
    return MongoClient(
        "mongodb+srv://12210104gcit:OJMEol5OXoHSuvwK@cluster0.zgrg8xm.mongodb.net/"
        "?retryWrites=true&w=majority&appName=Cluster0"
    )

client = get_mongo_client()
db = client["plant_disease_db"]
collection = db["soil_data"]
rec_collection = db["recommendation"]
health_collection = db["plant_health"]

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("best_model.h5")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

def preprocess_image(image, target_size=(128, 128)):
    image = image.convert("RGB").resize(target_size)
    return np.expand_dims(np.array(image) / 255.0, axis=0)

def predict_image(image, model, label_encoder):
    processed = preprocess_image(image)
    pred = model.predict(processed)[0]
    class_idx = np.argmax(pred)
    return label_encoder.classes_[class_idx]

def get_recommendation(label):
    try:
        doc = rec_collection.find_one({"label": label})
        if doc and "recommendation" in doc:
            return doc["recommendation"]
        return f"No recommendation found for {label}"
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return "Error loading recommendation"

def display_fixed_size(image, size=(300, 300)):
    image.thumbnail(size, Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", size, "white")
    new_image.paste(
        image,
        ((size[0] - image.size[0]) // 2, (size[1] - image.size[1]) // 2),
    )
    return new_image


def get_stress_data():
    try:
        # Query all documents with High or Moderate Stress
        cursor = health_collection.find({
            "Plant_Health_Status": {"$in": ["High Stress", "Moderate Stress"]}
        })
        docs = list(cursor)
        
        if not docs:
            st.warning("No stress data found.")
            return None
        
        # Randomly select one document
        chosen_doc = random.choice(docs)
        return chosen_doc

    except Exception as e:
        st.error(f"Error fetching stress data: {e}")
        return None

disease_descriptions = {
    "apple apple scab": "A fungal disease caused by Venturia inaequalis, leading to dark, scabby lesions on apple leaves and fruit.",
    "apple black rot": "Caused by Botryosphaeria obtusa, this disease results in fruit rot, leaf spots, and limb cankers on apple trees.",
    "apple healthy": "Leaves of apple plants showing no visible signs of disease or damage.",
    "corn (maize) common rust": "A fungal infection by Puccinia sorghi, causing reddish-brown pustules on leaves, which can reduce yield.",
    "corn (maize) northern leaf blight": "A leaf disease caused by Exserohilum turcicum, characterized by long, gray-green lesions that may kill leaves prematurely.",
    "corn (maize) healthy": "Normal corn leaves with uniform color and no disease symptoms.",
    "grape black rot": "Caused by Guignardia bidwellii, this disease produces dark, sunken spots on leaves and fruit rot on grapes.",
    "grape esca (black measles)": "A complex fungal disease resulting in interveinal discoloration, \"tiger-striped\" leaves, and dried berries.",
    "grape healthy": "Grape leaves in their natural, undiseased state.",
    "potato early blight": "A common fungal disease caused by Alternaria solani, showing concentric ring patterns on older leaves.",
    "potato late blight": "Caused by Phytophthora infestans, this is a devastating disease marked by dark, water-soaked lesions on leaves and tubers.",
    "potato healthy": "Potato leaves with no visible signs of blight or other infections.",
    "tomato bacterial spot": "Caused by Xanthomonas spp., this disease creates small, dark, water-soaked lesions on leaves and fruit.",
    "tomato early blight": "A fungal disease from Alternaria solani, leading to dark spots with concentric rings on lower leaves.",
    "tomato healthy": "Tomato leaves are in good condition, free from disease or pest damage."
}

def clean_label(raw_label):
    label = raw_label.replace("___", " ").replace("_", " ").lower().strip()
    label = " ".join(label.split())
    return label

with st.expander("Analysis on Dataset", expanded=True):
    st.subheader("Class Distribution")
    cols = st.columns([1, 6, 1])
    with cols[1]:
        st.image(Image.open("imagecount.png"), caption="Image Count per Class", width=1000)

    st.subheader("Mean RGB for Class")
    cols = st.columns([1, 6, 1])
    with cols[1]:
        st.image(Image.open("meanrgb.jpeg"), caption="RGB Mean per Class", width=1000)

    st.subheader("Analysis on Edges using OpenCV")

    for img_file, caption in [("healthy.jpeg", "Healthy Apple"), ("scab.jpeg", "Apple Scab"), ("blackrot.jpeg", "Apple Black Rot")]:
        cols = st.columns([1, 6, 1])
        with cols[1]:
            st.image(Image.open(img_file), caption=caption, width=1000)

with st.expander("Model Analysis", expanded=True):
    st.markdown("""
    ### Model Description and Limitations
    This model supports classification for **5 plants** and these classes:

    - Apple Apple scab
    - Apple Black rot
    - Apple healthy
    - Corn (maize) Common rust
    - Corn (maize) Northern Leaf Blight
    - Corn (maize) healthy
    - Grape Black rot
    - Grape Esca (Black Measles)
    - Grape healthy
    - Potato Early blight
    - Potato Late blight
    - Potato healthy
    - Tomato Bacterial spot
    - Tomato Early blight
    - Tomato healthy

    *Upload leaf images only from these plants and diseases.*
    """)

    model, label_encoder = load_assets()
    uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            display_img = display_fixed_size(image)
            st.image(display_img, caption="Uploaded Image", width=300)

            if st.button("🔎 Run Prediction"):
                with st.spinner("Analyzing image..."):
                    raw_label = predict_image(image, model, label_encoder)
                    st.success(f"**Prediction:** {raw_label}")

                    label = clean_label(raw_label)

                    if label in disease_descriptions:
                        st.markdown("### 🌱 Disease Description")
                        st.info(disease_descriptions[label])
                    else:
                        st.info("No description available for this label.")

                    st.subheader("Soil condition and Plant environment")
                    stress_data = get_stress_data()

                    if stress_data:
                        raw_features = [
                            "Soil_Moisture", "Ambient_Temperature", "Soil_Temperature", "Humidity",
                            "Light_Intensity", "Soil_pH", "Nitrogen_Level", "Phosphorus_Level",
                            "Potassium_Level", "Chlorophyll_Content", "Electrochemical_Signal"
                        ]

                        display_names = {
                            "Soil_Moisture": "Soil Moisture",
                            "Ambient_Temperature": "Ambient Temperature",
                            "Soil_Temperature": "Soil Temperature",
                            "Humidity": "Humidity",
                            "Light_Intensity": "Light Intensity",
                            "Soil_pH": "Soil pH",
                            "Nitrogen_Level": "Nitrogen Level",
                            "Phosphorus_Level": "Phosphorus Level",
                            "Potassium_Level": "Potassium Level",
                            "Chlorophyll_Content": "Chlorophyll Content",
                            "Electrochemical_Signal": "Electrochemical Signal"
                        }

                        stress_df = pd.DataFrame([{key: stress_data[key] for key in raw_features}])
                        display_df = stress_df.rename(columns=display_names).T.rename(columns={0: "Stress Record"})

                        healthy_docs = list(health_collection.find({"Plant_Health_Status": "Healthy"}))
                        if healthy_docs:
                            healthy_df = pd.DataFrame(healthy_docs)
                            healthy_mean = healthy_df[raw_features].mean()
                            comparison_df = pd.DataFrame({
                                "Current Record": stress_df.iloc[0],
                                "Healthy Mean": healthy_mean,
                                "Difference": stress_df.iloc[0] - healthy_mean
                            }).rename(index=display_names)


                            col1, col2 = st.columns([1, 1])  # Two equal-width columns

                            with col1:
                                st.dataframe(comparison_df.style.format("{:.2f}"))

                            with col2:
                                fig, ax = plt.subplots(figsize=(8, 5))
                                ax.plot(comparison_df.index, comparison_df["Current Record"], marker='o', label='Current Record')
                                ax.plot(comparison_df.index, comparison_df["Healthy Mean"], marker='o', label='Healthy Mean')
                                ax.set_ylabel("Values")
                                ax.set_title("Current Record Record vs Healthy Mean")
                                ax.legend()
                                plt.xticks(rotation=45, ha='right')
                                plt.grid(True)
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.info("No healthy records available for comparison.")
                    else:
                        st.info("No high or moderate stress record found.")

                    st.subheader("💊 Treatment Recommendation")
                    recommendation = get_recommendation(raw_label)
                    st.info(recommendation)

        except Exception as e:
            st.error(f"Error: {str(e)}")
