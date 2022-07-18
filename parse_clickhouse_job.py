import sys
from pyspark.sql import SparkSession

JAR_DIR = "jars/clickhouse-jdbc-0.3.1.jar"

def _spark_session():
    spark = SparkSession.builder \
    .config('spark.driver.extraClassPath', JAR_DIR) \
    .appName('ParseClickhousePySparkJob') \
    .getOrCreate()
    return spark

def make_query(first_date, last_date):
    return f"""(SELECT * FROM ads_data WHERE date BETWEEN '{first_date}' AND '{last_date}')"""

def parse(spark, query, first_date, last_date):
    df = spark.read \
        .format('jdbc') \
        .option("url", "jdbc:clickhouse://ds1.int.stepik.org:9010") \
        .option('dbtable', query) \
        .option("user", "readonly") \
        .option("password", "9be2980b6c1647c7998d1462af87c585") \
        .load()
    print('Spark succesfully read data from ClickHouse')

    df.write.parquet(f'raw_data/{last_date}')
    print('Spark saved data to folder "data"')

def main(argv):
    first_date = argv[0]
    last_date = argv[1]
    print('First date is ' + first_date)
    print('Last date is ' + last_date)

    spark = _spark_session()
    query = make_query(first_date, last_date)
    parse(spark, query, first_date, last_date)

    spark.stop()
    print('ClickHouse parsing succesfully completed')

if __name__ == '__main__':
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit('First and last dates are required')
    else:
        main(arg)
