from sklearn.metrics import accuracy_score
import model_repository

def evaluate_model(X_test, y_test):
    model = model_repository.load_model("titanic_rf_model")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    return accuracy