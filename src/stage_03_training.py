import argparse
import os
import numpy as np
import logging
from src.utils.common import read_yaml, create_directories
import random
from sklearn.ensemble import RandomForestClassifier
import joblib


STAGE = "STAGE_NAME" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):

    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts_content=config.get("artifacts")
    feature_dir_path=os.path.join(artifacts_content.get("ARTIFACTS_DIR"),artifacts_content.get("FEATURIZED_DATA"))
    feature_train_data_path=os.path.join(feature_dir_path,artifacts_content.get("FEATURIZED_DATA_TRAIN"))

    model_dir=os.path.join(artifacts_content.get("ARTIFACTS_DIR"),artifacts_content.get("MODEL_DIR"))
    create_directories([model_dir])
    model_path=os.path.join(model_dir,artifacts_content.get("MODEL_NAME"))

    matrix=joblib.load(feature_train_data_path)
    label=np.squeeze(matrix[:,1].toarray())
    X=matrix[:,2:]

    train_content=params.get("train")
    seed=train_content.get("seed")
    n_estimators=train_content.get("n_estimators")
    min_split=train_content.get("min_split")
    n_jobs=train_content.get("n_jobs")

    model=RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_split,
        n_jobs=n_jobs,
        random_state=seed
    )
    model.fit(X,label)
    joblib.dump(model,model_path)

    logging.info(f"input mat size {matrix.shape}")
    logging.info(f"input X size {X.shape}")
    logging.info(f"input label size {label.shape}")
    logging.info(f"succesfully save model {model_path}")




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