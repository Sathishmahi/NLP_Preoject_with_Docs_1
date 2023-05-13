import argparse
import os
import shutil
from urllib.request import urlretrieve
from pathlib import Path
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.data_management import process_posts
import random


STAGE = "TRAIN_TEST_XML2TSV" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def download_data(url:str,file_path:Path):
    create_directories(path_to_directories=[os.path.split(file_path)[0]])
    urlretrieve(url=url,filename=file_path)
def main(config_path, params_path):

    config = read_yaml(config_path)
    params = read_yaml(params_path)

    source_data_content=config.get("source_data")
    data_url=source_data_content.get("data_url")
    data_dir_name=source_data_content.get("data_dir")
    data_file_name=source_data_content.get("data_file")
    file_path=os.path.join(data_dir_name,data_file_name)
    download_data(url=data_url,file_path=file_path)

    prepare_data_content=params.get('prepare_data')
    split=prepare_data_content.get("split") 
    seed=prepare_data_content.get("seed") 
    random.seed(seed)

    artifacts_content=config.get("artifacts")
    preapred_data_dir=os.path.join(artifacts_content.get("ARTIFACTS_DIR")
                                    ,artifacts_content.get("PREPARED_DATA"))
    create_directories(path_to_directories=[Path(preapred_data_dir)])

    train_data_path=os.path.join(preapred_data_dir,artifacts_content.get('TRAIN_DATA'))
    test_data_path=os.path.join(preapred_data_dir,artifacts_content.get('TEST_DATA'))

    encode="utf8"
    with open(file=file_path,encoding=encode) as fd_in:
        with open(file=train_data_path,encoding=encode,mode="w") as fd_out_train:
            with open(file=test_data_path,encoding=encode,mode="w") as fd_out_test:
                process_posts(fd_in,fd_out_train,fd_out_test,"<python>",split)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e