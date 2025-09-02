import streamlit as st
import joblib as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data=pd.read_csv('app/data.csv')
    
    data=data.drop(['Unnamed: 32', 'id'],axis=1)
    data['diagnosis']=data['diagnosis'].map({'M':1, 'B':0})
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    
    input_dict = {}
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.text_input(
            label,
            value=str(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    x = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    
    for key, value in input_dict.items():
        value = float(value)  # Convert input value to float
        max_value = x[key].max()
        min_value = x[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_bar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    categories = ['Radius','Texture','Perimeter',
              'Area', 'Smoothness','Compactness','Concavity',
              'Concave Points','Symmetry','Fractal Dimention']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        name='Mean'
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
            input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        name='Standard Error'
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        name='Worst Value'
    ))

    fig.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        title='Comparison of Features',
        xaxis=dict(title='Feature'),
        yaxis=dict(title='Scaled Value')
    )
    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1,-1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.image("app/Benign.png", width=200)  # Add image for benign prediction
        st.success("The mass is predicted to be **benign**.")
    else:
        st.image("app/Maliginant.png", width=200)  # Add image for malignant prediction
        st.error("The mass is predicted to be **malignant**.")
    
    st.subheader("Prediction Details:")
    st.write("Here are some additional details about the prediction:")
    probas = model.predict_proba(input_array_scaled)
    st.write("Probability of being Benign: ", probas[0][0])
    st.write("Probability of being Malignant: ", probas[0][1])

    st.info("This prediction is for informational purposes only and should not replace professional medical advice.")

    # Add some style to the output
    st.markdown(""" 
    <style> 
    .st-bb {
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .st-bb:hover {
        background-color: #f0f0f0;
    }
    </style> 
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    input_data = add_sidebar()
    with st.container():
        st.title("Cancer Predictor")
        st.write("This app is designed to interface with your cytology lab, allowing for seamless integration of tissue sample data. By utilizing a machine learning model, the app can accurately predict whether a mass is benign or malignant based on the measurements provided by the lab. Additionally, users have the flexibility to manually adjust measurements using the sliders available in the sidebar, ensuring the most accurate diagnosis possible.")
    
    # row1, row2=st.rows([1,1])
    
    # with row1:
        rader_chart = get_bar_chart(input_data)
        st.plotly_chart(rader_chart)
    # with row2:
        add_predictions(input_data)
        # st.write(model_type)
        st.write("This is not used for professional doctors.")
        
        
if __name__=='__main__':
    main()
