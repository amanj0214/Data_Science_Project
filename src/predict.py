import joblib
import config


def batch_prediction(model=config.LATEST_MODEL_FILE):
    ...


def predict(df_to_predict, model_file=config.LATEST_MODEL_FILE):
    model = joblib.load(model_file)
    return model.predict(df_to_predict)


if __name__ == "__main__":
    batch_prediction()
