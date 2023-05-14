import os
import yaml
import logging
import time
import pandas as pd
import pickle
import json
import numpy as np
from pathlib import Path
import scipy.sparse as sparse
import  joblib 

def get_df(file_path:Path,column_names:list,sep:str="\t",
            header=None,encoding="utf8")->pd.DataFrame:
    df = pd.read_csv(file_path,sep=sep,encoding=encoding)
    logging.info(f'input dataframe path {file_path}  data frame size is {df.shape}')
    return df
def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")

def save_matrix(df, text_matrix, out_path):
    pid_matrix = sparse.csr_matrix(df.iloc[:,0].astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.iloc[:,2].astype(np.int64)).T

    result = sparse.hstack([pid_matrix, label_matrix, text_matrix], format="csr")

    msg = f"The output matrix saved at {out_path} of shape: {result.shape}"
    logging.info(msg)
    joblib.dump(result, out_path) 

def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")