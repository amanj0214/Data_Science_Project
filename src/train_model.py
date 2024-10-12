import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pyspark.sql import SparkSession
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import DMatrix
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier
import model_repository
import pandas as pd
import numpy as np
from sqlalchemy import text

def train_model(engine, large_data=False, use_spark=False):
    if use_spark:
        spark = SparkSession.builder.getOrCreate()
        data = load_data_spark(spark, engine)
    else:
        query = "SELECT * FROM titanic_processed_data"
        data = pd.read_sql(text(query), engine)
    
    X = data.drop(columns=['Survived'])
    y = data['Survived']
    
    # Train-test split for small data or use PySpark DataFrames
    if use_spark:
        return train_spark_model(data)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if large_data:
            return train_xgboost_with_hyperopt(X_train, X_test, y_train, y_test, X, y)
        else:
            return train_sklearn_with_gridsearch(X_train, X_test, y_train, y_test, X, y)

def load_data_spark(spark, engine):
    query = "(SELECT * FROM titanic_processed_data) AS data"
    data = spark.read.format("jdbc").options(
        url="jdbc:sqlite://{}".format(engine.url),
        dbtable=query
    ).load()
    return data

def train_sklearn_with_gridsearch(X_train, X_test, y_train, y_test, X, y):
    model_params = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'criterion': ['gini', 'entropy']
            }
        }
    }
    
    best_model = None
    best_params = None
    best_score = 0
    
    # Scikit-learn GridSearch
    for model_name, mp in model_params.items():
        grid = GridSearchCV(mp['model'], mp['params'], cv=5)
        grid.fit(X_train, y_train)
        
        with mlflow.start_run(run_name=f"{model_name}_GridSearch"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("best_score", grid.best_score_)
            
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_params = grid.best_params_
    
    # Retrain on full data
    best_model.fit(X, y)
    with mlflow.start_run(run_name="Best_Model_Retrain"):
        mlflow.sklearn.log_model(best_model, "best_model_retrained")
        mlflow.log_params(best_params)
        mlflow.log_metric("retrained_on_full_data", True)
        model_repository.save_model(best_model, "titanic_rf_model_final")
    
    return best_model, X_test, y_test

def train_xgboost_with_hyperopt(X_train, X_test, y_train, y_test, X, y):
    def objective(params):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        with mlflow.start_run(run_name="XGBoost_Hyperopt"):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
        
        return {'loss': -accuracy, 'status': STATUS_OK}
    
    space = {
        'max_depth': hp.choice('max_depth', range(3, 10)),
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
        'gamma': hp.uniform('gamma', 0.0, 0.5),
        'subsample': hp.uniform('subsample', 0.5, 1.0)
    }
    
    trials = Trials()
    best_params = fmin(fn=objective,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=20,
                       trials=trials)
    
    # Retrain best model on full dataset
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X, y)
    
    with mlflow.start_run(run_name="Best_XGBoost_Retrain"):
        mlflow.xgboost.log_model(best_model, "best_xgboost_model_retrained")
        mlflow.log_params(best_params)
        mlflow.log_metric("retrained_on_full_data", True)
        model_repository.save_model(best_model, "titanic_best_xgboost_model_final")
    
    return best_model, X_test, y_test

def train_spark_model(data):
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
    output = assembler.transform(data)
    
    train, test = output.randomSplit([0.8, 0.2])
    
    gbt = GBTClassifier(featuresCol="features", labelCol="Survived")
    
    paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]) \
                                  .addGrid(gbt.maxIter, [50, 100]).build()
    
    crossval = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=5)
    
    cv_model = crossval.fit(train)
    
    predictions = cv_model.transform(test)
    
    best_model = cv_model.bestModel
    with mlflow.start_run(run_name="Spark_GBT_Model"):
        mlflow.spark.log_model(best_model, "best_spark_gbt_model")
        mlflow.log_metric("accuracy", BinaryClassificationEvaluator().evaluate(predictions))
    
    return best_model, train, test
