import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

HOME_PATH = os.getcwd()
DATA_PATH = os.path.join(HOME_PATH, 'inputs')
MODELS_PATH = os.path.join(HOME_PATH, 'models')

st.write("""
# Penguin Classifier

Predict the **Palmer Penguin** species
""")

st.sidebar.header("Input Features:")

uploaded_file = st.sidebar.file_uploader("Upload your CSV input file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df.drop(columns=["species"], inplace=True)
else:
    def user_input_features():
        """

        :return:
        """

        island = st.sidebar.selectbox(
            "Island", (
                "Biscoe",
                "Dream",
                "Torgersen"
            )
        )

        sex = st.sidebar.selectbox(
            "Sex", (
                "male",
                "female"
            )
        )

        bill_length_mm = st.sidebar.slider(
            "Bill length (mm)", 32.1, 59.6, 43.9
        )

        bill_depth_mm = st.sidebar.slider(
            "Bill depth (mm)", 13.1, 21.5, 17.2
        )

        flipper_length_mm = st.sidebar.slider("Flipper length (mm)", 172.0, 231.0, 201.0)

        body_mass_g = st.sidebar.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)

        data = {
            "island": island,
            "sex": sex,
            "bill_length_mm": bill_length_mm,
            "bill_depth_mm": bill_depth_mm,
            "flipper_length_mm": flipper_length_mm,
            "body_mass_g": body_mass_g
        }

        features = pd.DataFrame(data, index=[0])

        return features

    input_df = user_input_features()


penguins_raw = pd.read_csv(os.path.join(DATA_PATH, "penguins_cleaned.csv"))
penguins = penguins_raw.drop(columns=["species"])
df = pd.concat([input_df, penguins], axis=0)

encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df.sample(10)

st.subheader("User Input features")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded.")
    st.write(df)

load_clf = pickle.load(open(os.path.join(MODELS_PATH, "penguins_clf.pkl"), "rb"))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(["Adelie", "Chinstrap", " Gentoo"])
st.write(penguins_species[prediction])

st.subheader("Prediction Probability")
st.write(prediction_proba)
