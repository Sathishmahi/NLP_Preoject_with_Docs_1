import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories,save_json
import random
import joblib
from sklearn.metrics import accuracy_score,confusion_matrix,average_precision_score,roc_curve,roc_auc_score,precision_recall_curve
import numpy as np


STAGE = "STAGE_04_EVALUATION" ## <<< change stage name 

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
    feature_dir_path=os.path.join(artifacts_content.get("ARTIFACTS_DIR"),artifacts_content.get("FEATURIZED_DATA"))
    feature_test_data_path=os.path.join(feature_dir_path,artifacts_content.get("FEATURIZED_DATA_TEST"))

    model_dir=os.path.join(artifacts_content.get("ARTIFACTS_DIR"),artifacts_content.get("MODEL_DIR"))
    model_path=os.path.join(model_dir,artifacts_content.get("MODEL_NAME"))

    matrix=joblib.load(feature_test_data_path)
    label=np.squeeze(matrix[:,1].toarray())
    print(set(label))
    X=matrix[:,2:]

    score_content=config.get("plots")
    PRC_json_path=score_content.get("PRC")
    AUC_json_path=score_content.get("ROC")

    model=joblib.load(model_path)
    predicted_out=model.predict(X)
    predicted_out_prob=model.predict_proba(X)
    score=model.score(X,label)
    roc_auc=roc_auc_score(label,predicted_out)

    average_precision=average_precision_score(label,predicted_out)

    precision,recall,prc_thersold=precision_recall_curve(label,predicted_out_prob[:,1])

    min,max=1,5
    n_th=random.randint(min, max)
    prc_list=list(zip(precision,recall,prc_thersold))[::n_th]

    prc_dict={
        "prc":[
            {"precioson":p,"recall":r,"thersold":t}
            for p,r,t in prc_list
        ]
    }

    tp,fp,thersold=roc_curve(label,predicted_out_prob[:,1])

    roc_dict={
        "prc":[
            {"true_positive":t,"false_positive":f,"thersold":th}
            for t,f,th in zip(tp,fp,thersold)
        ]
    }


    json_path=config.get("metrics").get("SCORES")

    json_content={"score":score,"average_precision":average_precision,"roc_auc":roc_auc}
    save_json(path=json_path, data=json_content)
    save_json(path=PRC_json_path, data=prc_dict)
    save_json(path=AUC_json_path, data=roc_dict)

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