
import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd

st.markdown(
    """
    <style>
    .stApp {
        background-color: #90EE90;
        color: brown;   /* default text color */
    }

    h1 {
        color: #1f77b4;  /* blue title */
    }

    label {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label span p {
        color: brown !important;
        font-weight: 600;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


le = pickle.load(open("le_bean_classifier.pkl", "rb"))   # LabelEncoder
pt = pickle.load(open("powertransformer_bean_classifier_org.pkl", "rb"))
scaler = pickle.load(open("scaler_bean_classifier_org.pkl", "rb"))
model = pickle.load(open("best_model_bean_classifier_org.pkl", "rb")) 

st.title("Dry Bean Type Classification")
st.image("Bean_Img_PPT.jpg", 
             #caption="Income Classifier", 
              use_container_width=True)

st.write("This app Classifies the type of dry bean based on the features")

area = st.number_input("Area", min_value=0, value=0, step=1)
perimeter = st.number_input("Perimeter", min_value=0.0, step=0.001,format="%.3f")
major_axis_length = st.number_input("Major Axis Length",min_value=0.0,value=0.0,step=0.000001,format="%.6f")
minor_axis_length = st.number_input("Minor Axis Length", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
eccentricity = st.number_input("Eccentricity", min_value=0.0,value=0.0,step=0.000001,format="%.6f")        
convex_area = st.number_input("Convex Area", min_value=0, value=0, step=1)
equivalent_diameter = st.number_input("Equivalent Diameter", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
extent = st.number_input("Extent", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
solidity = st.number_input("Solidity", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
roundness = st.number_input("Roundness", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
compactness = st.number_input("Compactness", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
shape_factor_1 = st.number_input("Shape Factor 1", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
shape_factor_2 = st.number_input("Shape Factor 2", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
shape_factor_3 = st.number_input("Shape Factor 3", min_value=0.0,value=0.0,step=0.000001,format="%.6f")
shape_factor_4 = st.number_input("Shape Factor 4", min_value=0.0,value=0.0,step=0.000001,format="%.6f")        


if st.button("Classify Bean"):
    input_data = pd.DataFrame({
        "area": [area],
        "perimeter": [perimeter],
        "majoraxislength": [major_axis_length],
        "minoraxislength": [minor_axis_length],
        "aspectratio": [aspect_ratio],
        "eccentricity": [eccentricity],
        "convexarea": [convex_area],
        "equivdiameter": [equivalent_diameter],
        "extent": [extent],
        "solidity": [solidity],
        "roundness": [roundness],
        "compactness": [compactness],
        "shapefactor1": [shape_factor_1],
        "shapefactor2": [shape_factor_2],
        "shapefactor3": [shape_factor_3],
        "shapefactor4": [shape_factor_4]
    })

    # If user entered all zeros, prompt for correct values
    if input_data.eq(0).all().all():
        #st.error("Please enter non-zero values for the features before classifying.")
        st.markdown(
        "<p style='color:brown; font-weight:600;'>Please enter non-zero values for the features before classifying.</p>",
        unsafe_allow_html=True
        )
        st.stop()

    skewed_cols = ['area',
        'perimeter',
        'majoraxislength',
        'minoraxislength',
        'eccentricity',
        'convexarea',
        'equivdiameter',
        'extent',
        'solidity',
        'shapefactor2',
        'shapefactor4']
    
    input_data[skewed_cols] = pt.transform(input_data[skewed_cols])
    feature_order = ['area', 'perimeter', 'majoraxislength', 'minoraxislength',
         'aspectratio', 'eccentricity', 'convexarea', 'equivdiameter', 'extent',
         'solidity', 'roundness', 'compactness', 'shapefactor1', 'shapefactor2',
         'shapefactor3', 'shapefactor4']
    
    input_data = input_data[feature_order]
    input_data_scaled = scaler.transform(input_data)    
    prediction = model.predict(input_data_scaled)
    predicted_label = le.inverse_transform(prediction)[0]   

    st.markdown(
            f"""
            <div style="
                background-color:#e8f0ff;
                padding:15px;
                border-radius:10px;
                border:2px solid #0B3D91;
                color:#0B3D91;
                font-size:18px;
                font-weight:bold;
                text-align:center;
            ">
                Predicted Dry Bean Type: {predicted_label}
           
            </div>
            """,
            unsafe_allow_html=True
        )