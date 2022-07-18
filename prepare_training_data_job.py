import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2

SEED = 42

def _spark_session():
    spark = SparkSession.builder \
    .appName('PrepareTrainingDataPySparkJob') \
    .getOrCreate()
    return spark

def get_features(df):
    """Функция для создания признаков, на которых будут обучаться модели"""
    # Создаем признаки is_cpm и is_cpc
    ndf = (df
           .withColumn('is_cpm', F.when(F.col('ad_cost_type') == 'CPM', 1).otherwise(0))
           .withColumn('is_cpc', F.when(F.col('ad_cost_type') == 'CPC', 1).otherwise(0)))
    
    # Создаем вспомогательные признаки is_view и is_click
    # Они понадобятся для расчета суммарных просмотров и кликов на объявления
    ndf = (ndf
           .withColumn('is_view', F.when(F.col('event') == 'view', 1).otherwise(0))
           .withColumn('is_click', F.when(F.col('event') == 'click', 1).otherwise(0)))
      
    # Группируем данные по id объявлений
    features = (ndf.groupBy('ad_id').agg(
        F.max(F.col('target_audience_count')).alias('target_audience_count'),
        F.max(F.col('has_video')).alias('has_video'),
        F.max(F.col('is_cpm')).alias('is_cpm'),
        F.max(F.col('is_cpc')).alias('is_cpc'),
        F.max(F.col('ad_cost')).alias('ad_cost'),
        F.sum(F.col('is_view')).alias('views_count'), # Суммируя, получим число просмотров.
        F.sum(F.col('is_click')).alias('clicks_count'), # Суммируя, получим число кликов.
        F.countDistinct(F.col('date')).alias('days_count')))

    # Считаем CTR и избавляемся от вспомогательных колонок views_count и clicks_count
    features = (features.withColumn('ctr', F.col('clicks_count') / F.col('views_count'))
                        .drop('views_count', 'clicks_count'))
    return features

def process(spark, parse_date):
    df = spark.read.parquet(f'raw_data/{parse_date}')

    # С помощью самодельной функции создаем нужные фичи из исходного датафрейма
    features = get_features(df)
    
    # Разбиваем данные на обучающую и тестовую выборки
    train_df, test_df = features.randomSplit([TRAIN_SIZE, TEST_SIZE], seed=SEED)
    
    # Сохраняем получившиеся выборки
    train_path = f'data/{parse_date}/train'
    test_path = f'data/{parse_date}/test'

    train_df.write.parquet(train_path)
    test_df.write.parquet(test_path)

def main(argv):
    parse_date = argv[0]
    print('Parse date is ' + parse_date)

    spark = _spark_session()
    process(spark, parse_date)

    spark.stop()
    print(f'Train and test for {parse_date} are prepared and saved to data/{parse_date}')


if __name__ == '__main__':
    arg = sys.argv[1:]
    if len(arg) != 1:
        sys.exit('Parse date is required')
    else:
        main(arg)
