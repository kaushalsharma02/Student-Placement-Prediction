import streamlit as st
import pickle
import numpy as np
import pandas as pd


# ---------- Load files ----------

model = pickle.load(open("model.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))
cm = pickle.load(open("cm.pkl", "rb"))


# ---------- Page setting ----------

st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered"
)


# ---------- Sidebar ----------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction", "Model Performance"]
)


# ---------- HOME ----------

if page == "Home":

    st.title("🎓 Student Placement Prediction")

    st.write(
        "This Machine Learning project predicts whether a student "
        "will be placed or not using Logistic Regression."
    )

    df = pd.read_csv("collegePlace.csv")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


# ---------- PREDICTION ----------

elif page == "Prediction":

    st.title("Placement Prediction")

    # ----- mapping -----

    stream_map = {
        "Electronics And Communication": 0,
        "Computer Science": 1,
        "Information Technology": 2,
        "Mechanical": 3,
        "Electrical": 4,
        "Civil": 5
    }

    gender_map = {
        "Male": 0,
        "Female": 1
    }

    hostel_map = {
        "No": 0,
        "Yes": 1
    }

    # ----- input -----

    gender_name = st.selectbox("Gender", list(gender_map.keys()))

    age = st.number_input("Age", 18, 30)

    stream_name = st.selectbox(
        "Stream",
        list(stream_map.keys())
    )

    internships = st.number_input("Internships", 0, 10)

    
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=None,
        placeholder="Enter CGPA"
    )

    hostel_name = st.selectbox(
        "Hostel",
        list(hostel_map.keys())
    )

    backlogs = st.number_input("Backlogs", 0, 10)

    # ----- convert to number -----

    gender = gender_map[gender_name]
    stream = stream_map[stream_name]
    hostel = hostel_map[hostel_name]


    if st.button("Predict"):

        data = np.array([[
            gender,
            age,
            stream,
            internships,
            cgpa,
            hostel,
            backlogs
        ]])

        result = model.predict(data)

        if result[0] == 1:
            st.success("Student will be Placed")
        else:
            st.error("Student will NOT be Placed")

# ---------- PERFORMANCE ----------

elif page == "Model Performance":

    st.title("Model Performance")

    st.write("Accuracy : ", round(accuracy * 100, 2), "%")

    st.subheader("Confusion Matrix")

    st.write(cm)