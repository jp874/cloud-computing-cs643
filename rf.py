from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.param import Param, Params
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest


sc = SparkContext()
spark = SparkSession(sc)

inputDF = spark.read.csv('s3://practicaltraining/TrainingDataset.csv',header='true', inferSchema='true', sep=';')

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in inputDF.columns if c != 'quality']


transformed_df= inputDF.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))



model = RandomForest.trainClassifier(transformed_df,numClasses=10,categoricalFeaturesInfo={}, numTrees=500, maxBins=1280, maxDepth=30, seed=33)

model.save(sc,"s3://practicaltraining/trainingmodel")



