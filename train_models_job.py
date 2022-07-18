import sys

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor

SEED = 42

def _spark_session():
    spark = SparkSession \
    .builder \
    .appName('TrainModelsPySparkJob') \
    .getOrCreate()
    return spark
    
    
def train_dt(train_data, features, evaluator): 
    print('Decision tree training starts')
    
    dt = DecisionTreeRegressor(featuresCol='features',
                               labelCol='ctr',
                               predictionCol='prediction',
                               seed=SEED)

    dt_grid = (ParamGridBuilder()
               .addGrid(dt.maxDepth, range(4, 11, 2))
               .build())

    dt_cv = CrossValidator(estimator=dt,
                           estimatorParamMaps=dt_grid,
                           evaluator=evaluator,
                           numFolds=3)

    dt_pipeline = Pipeline(stages=[features, dt_cv]).fit(train_data)
    print('Decision tree training completed successfully')
    return dt_pipeline
  
    
def train_rf(train_data, features, evaluator):
    print('Random forest training starts')
    
    rf = RandomForestRegressor(featuresCol='features',
                               labelCol='ctr',
                               predictionCol='prediction',
                               seed=SEED)

    rf_grid = (ParamGridBuilder()
               .addGrid(rf.maxDepth, range(4, 11, 2))
               .addGrid(rf.numTrees, range(1, 21, 10))
               .build())

    rf_cv = CrossValidator(estimator=rf,
                           estimatorParamMaps=rf_grid,
                           evaluator=evaluator,
                           numFolds=3)

    rf_pipeline = Pipeline(stages=[features, rf_cv]).fit(train_data)
    print('Random forest training completed successfully')
    return rf_pipeline


def train_gbt(train_data, features, evaluator):
    print('Gradient boosting training starts')
    
    gbt = GBTRegressor(featuresCol='features',
                       labelCol='ctr',
                       predictionCol='prediction',
                       seed=SEED)

    gbt_grid = (ParamGridBuilder()
                .addGrid(gbt.maxDepth, range(2, 11, 4))
                .addGrid(gbt.stepSize, [0.03, 0.1])
                .build())

    gbt_cv = CrossValidator(estimator=gbt,
                            estimatorParamMaps=gbt_grid,
                            evaluator=evaluator,
                            numFolds=3)
    
    gbt_pipeline = Pipeline(stages=[features, gbt_cv]).fit(train_data)
    print('Gradient boosting training completed successfully')
    return gbt_pipeline


def process(spark, data_date):
    # Загружаем данные
    train_path = f'data/{data_date}/train'
    test_path = f'data/{data_date}/test'

    train_data = spark.read.parquet(train_path)
    test_data = spark.read.parquet(test_path)
    
    # Готовим фичи
    feature_cols = list(set(train_data.columns) - {'ad_id', 'ctr'})
    features = VectorAssembler(inputCols=feature_cols, outputCol='features')

    # Создаем оценщика, который будет рассчитывать RMSE моделей
    evaluator = RegressionEvaluator(labelCol='ctr',
                                    predictionCol='prediction',
                                    metricName='rmse')
    
    # Подбираем лучшие параметры и обучаем модели
    print('Models training in progress...')
    dt_model = train_dt(train_data, features, evaluator)
    rf_model = train_rf(train_data, features, evaluator)
    gbt_model = train_gbt(train_data, features, evaluator)
    print('All models trained successfully')
    
    # Оцениваем качество лучших моделей каждого вида
    dt_rmse = evaluator.evaluate(dt_model.transform(test_data))
    rf_rmse = evaluator.evaluate(rf_model.transform(test_data))
    gbt_rmse = evaluator.evaluate(dt_model.transform(test_data))
    
    # Ищем модель с минимальным RMSE
    rmse_model_dict = {
        dt_rmse: dt_model,
        rf_rmse: rf_model,
        gbt_rmse: gbt_model
    }
    best_model = rmse_model_dict[min(rmse_model_dict.keys())]

    # Сохраняем лучшую модель
    best_model_path = f'best_models/{data_date}'
    best_model.write().overwrite().save(best_model_path)
    print('Best model saved')
    
    spark.stop()    
    
def main(argv):
    data_date = argv[0]
    print(f'Models will be trained on data parsed on {data_date}')
    spark = _spark_session()
    process(spark, data_date)


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 1:
        sys.exit('Data parse date is required to find train and test data')
    else:
        main(arg)
