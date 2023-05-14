import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories,get_df,save_matrix
import random
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


STAGE = "STAGE FEATURIZATION" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts_content=config.get("artifacts")
    preapred_data_dir=os.path.join(artifacts_content.get("ARTIFACTS_DIR")
                                    ,artifacts_content.get("PREPARED_DATA"))

    train_data_path=os.path.join(preapred_data_dir,artifacts_content.get('TRAIN_DATA'))
    test_data_path=os.path.join(preapred_data_dir,artifacts_content.get('TEST_DATA'))

    featurize_content=params.get("featurize")
    no_of_features=featurize_content.get("no_of_features")
    n_grams=featurize_content.get("n_grams")
    column_names=featurize_content.get("column_names")

    feature_dir_path=os.path.join(artifacts_content.get("ARTIFACTS_DIR"),artifacts_content.get("FEATURIZED_DATA"))
    feature_train_data_path=os.path.join(feature_dir_path,artifacts_content.get("FEATURIZED_DATA_TRAIN"))
    feature_test_data_path=os.path.join(feature_dir_path,artifacts_content.get("FEATURIZED_DATA_TEST"))

    create_directories(path_to_directories=[feature_dir_path])

    df_train=get_df(file_path=train_data_path, column_names=column_names)
    df_test=get_df(file_path=test_data_path, column_names=column_names)

    train_text=df_train.text.str.lower().values.astype('U')
    bag_of_words=CountVectorizer(
        stop_words="english",
        max_features=no_of_features,
        ngram_range=(1,n_grams)
        )

    bag_of_words.fit(train_text)
    train_binary_matrix=bag_of_words.transform(train_text)

    # print(train_binary_matrix.shape)

    tfidf=TfidfTransformer(smooth_idf=False)
    tfidf.fit(train_binary_matrix)
    tfidf_train_matrix=tfidf.transform(train_binary_matrix)


    test_text=df_test.text.str.lower().values.astype('U')
    test_binary_matrix=bag_of_words.transform(test_text)
    tfidf_test_matrix=tfidf.transform(test_binary_matrix)
    print(tfidf_train_matrix.shape)
    print(tfidf_test_matrix.shape)

    train_pkl=os.path.join(feature_dir_path,artifacts_content.get("FEATURIZED_DATA_TRAIN"))
    test_pkl=os.path.join(feature_dir_path,artifacts_content.get("FEATURIZED_DATA_TEST"))

    save_matrix(df_train, tfidf_train_matrix, train_pkl)
    save_matrix(df_test, tfidf_test_matrix, test_pkl)


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