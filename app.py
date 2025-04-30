import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np

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


def desc_calc():
    """
    Calculates molecular descriptors using the PaDEL-Descriptor tool.
    This function runs a Java command-line program and removes the input file after processing.
    """
    # Command to run the PaDEL-Descriptor tool
    bashCommand = "java -Xms2G -Xmx2G ..."

    # Run the command and capture output/errors
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Clean up the input file
    os.remove('molecule.smi')



def filedownload(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="prediction.csv",
        mime="text/csv",
    )


# Configuration for file paths
MODEL_PATH = 'acetylcholinesterase_model.pkl'
DESCRIPTOR_LIST_PATH = 'descriptor_list.csv'
DESCRIPTORS_OUTPUT_PATH = 'descriptors_output.csv'

def build_model(input_data):
    # Load the model using the configured path
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            load_model = pickle.load(model_file)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the file is available.")
        return

    # Make predictions and display results
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)



# Logo image
image = Image.open('loogo.png')

st.image(image, use_container_width=True)

# Page title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app allows you to predict the bioactivity towards inhibting the `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for Alzheimer's disease.
""")




# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

def check_dependencies():
    # Check if Java is available
    if subprocess.call(["java", "-version"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) != 0:
        st.error("Java is not installed or not available in PATH.")
        st.stop()

    # Check if PaDEL-Descriptor JAR file exists
    if not os.path.exists('./PaDEL-Descriptor/PaDEL-Descriptor.jar'):
        st.error("PaDEL-Descriptor JAR file not found. Please ensure it is in the correct location.")
        st.stop()

    # Check if model file exists
    if not os.path.exists('acetylcholinesterase_model.pkl'):
        st.error("Model file not found. Please ensure it is in the correct location.")
        st.stop()

# Call the dependency check function at the start of the script
check_dependencies()


if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        try:
            # Validate and read the uploaded file
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            if load_data.empty:
                st.error("Uploaded file is empty. Please upload a valid input file.")
                st.stop()
            
            # Save the input data for descriptor calculation
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read and display calculated descriptors
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Subset descriptors for prediction
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('descriptor_list.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model
            build_model(desc_subset)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a file to proceed.")
