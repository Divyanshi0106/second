import streamlit as st
import pandas as pd
from PIL import Image
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

# Load logo
image = Image.open('loogo.png')
st.image(image, use_container_width=True)

# App title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app predicts the bioactivity of molecules as Acetylcholinesterase inhibitors — a key enzyme in Alzheimer's research.

---
""")

# RDKit descriptor calculator
def desc_calc_rdkit(smiles_list):
    desc_names = [desc[0] for desc in Descriptors._descList]
    data = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            data.append([np.nan] * len(desc_names))  # Handle invalid SMILES
            continue
        descriptors = [desc(mol) for _, desc in Descriptors._descList]
        data.append(descriptors)

    df = pd.DataFrame(data, columns=desc_names)
    return df

# File download function
def filedownload(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="prediction.csv",
        mime="text/csv",
    )

# Model prediction
def build_model(input_data, molecule_names):
    try:
        with open('acetylcholinesterase_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'acetylcholinesterase_model.pkl' is in the app folder.")
        return

    prediction = model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(molecule_names, name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    filedownload(df)

# Sidebar
with st.sidebar.header('1. Upload your input file'):
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file with SMILES and molecule name", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

# Main logic
if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        try:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            if load_data.empty:
                st.error("Uploaded file is empty.")
                st.stop()

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                smiles_list = load_data[0].tolist()
                desc = desc_calc_rdkit(smiles_list)

            st.header('**Calculated molecular descriptors**')
            st.write(desc)
            st.write(desc.shape)

            st.header('**Subset of descriptors from trained model**')
            Xlist = list(pd.read_csv('descriptor_list.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            build_model(desc_subset, load_data[1])

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a file to proceed.")
