# About the Project
TBD

# Folder Structure 🗂️
```
📦biosignals-gen-ai
 ┣ 📂assets                    <-- Contains saved figures, ...
 ┣ 📂config                    <-- Configuration files for the pipeline
 ┣ 📂data                      <-- Provided data
 ┃ ┣ 📂raw                     <-- Contains the raw data provided by the supervisor
 ┃ ┗ 📂processed               <-- Contains the processed data
 ┣ 📂models                    <-- Saved models during Development
 ┣ 📂notebooks                 <-- Jupyter Notebooks used in development
 ┣ 📂src                       <-- Source code / modules / classes
 ┣ 📂tests                     <-- Unit tests for the source code
 ┃ ┣ 📜dataloading.py          <-- Class that handles the data loading
 ┃ ┣ 📜preprocessing.py        <-- Class that handles the data preprocessing
 ┃ ┣ 📜utils.py                <-- Contains utility / helper functions
 ┃ ┣ 📜vae_base.py             <-- Abstract class of VAE
 ┃ ┣ 📜vae_dense.py            <-- Implementation of base VAE using Dense layers
 ┃ ┗ 📜vae_lstm.py             <-- Implementation of base VAE using LSTM layers
 ┣ 🕹️main.py                   <-- Entry point of the pipeline
 ┣ 📜README.md                 <-- The top-level README for developers using this project
 ┗ 📜requirements.txt          <-- The requirenments file for reproducing the environment
```

# Setting up the environment and run the code

1. Clone the repository by running the following command in your terminal:

   ```
   git clone https://github.com/negralessio/biosignals-gen-ai
   ```


2. Navigate to the project root directory by running the following command in your terminal:

   ```
   cd biosignals-gen-ai
   ```

3. [Optional] Create a virtual environment and activate it. For example, using the built-in `venv` module in Python:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages by running the following command in your terminal:

   ```
   pip install -r requirements.txt
   ```

5. Place the data in the `data/raw` folder.

6. Run the pipeline with the following command:

   ```
   python3 main.py --config "configs/config.yaml"