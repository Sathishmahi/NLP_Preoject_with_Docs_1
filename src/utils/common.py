import os
import yaml
import logging
import time
import pandas as pd
import pickle
import json
from pathlib import Path


def get_df(file_path:Path,column_names:list,sep:str="\t",
            header=None,encoding="utf8")->pd.DataFrame:
    df = pd.read_csv(file_path,sep=sep,header=header,encoding=encoding,names=column_names)
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

def to_save_pkl(file_paths:list,objs):
    if len(file_paths)!=len(objs):
        raise Exception("file paths and objs list must be a same len")
    for file_path,obj in zip(file_paths,objs): 
        with open(file_path,'wb') as pkl_file:
            pickle.dump(obj=obj, file=pkl_file)

def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")