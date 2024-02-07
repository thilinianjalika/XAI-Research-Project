import os
import joblib
from datasets import IMDBDataset
from models import SVCModel, RFModel, KNNModel, LRModel

if __name__ == "__main__":
    # Directory variables
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    intermediate_data_dir = os.environ["SM_OUTPUT_INTERMEDIATE_DIR"]
    model_output_dir = os.environ["SM_MODEL_DIR"]
    output_data_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    ds_config_path = f"{data_dir}/imdb.yaml"
    ds = IMDBDataset(ds_config_path, vectorizer_fitted=False)
    print("Dataset instantiated")

    rf_model = RFModel()
    svc_model = SVCModel()
    knn_model = KNNModel()
    lr_model = LRModel()
    print("Models instantiated")

    rf_model.fit(ds.x_train, ds.y_train)
    print("RF completed")
    svc_model.fit(ds.x_train, ds.y_train)
    print("SVC completed")
    knn_model.fit(ds.x_train, ds.y_train)
    print("KNN completed")
    lr_model.fit(ds.x_train, ds.y_train)
    print("LR completed")

    rf_model.save(f"{model_output_dir}/rf.pkl")
    svc_model.save(f"{model_output_dir}/svm.pkl")
    knn_model.save(f"{model_output_dir}/knn.pkl")
    lr_model.save(f"{model_output_dir}/lr.pkl")
    joblib.dump(ds.input_encoder, f"{model_output_dir}/tfidf.pkl")
    print("Models saved")
