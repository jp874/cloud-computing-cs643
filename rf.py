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
from pyspark.ml.classification import RandomForestClassifier
#from pyspark.mllib.classification import RandomForestClassifier
from pyspark.mllib.tree import RandomForest
sc = SparkContext()
spark = SparkSession(sc)

inputDF = spark.read.csv('s3://practicaltraining/t.csv',header='true', inferSchema='true', sep=';')

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in inputDF.columns if c != 'quality']

transformed_df= inputDF.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))



model = RandomForest.trainClassifier(transformed_df,numClasses=10,categoricalFeaturesInfo={}, numTrees=500, maxBins=1280, maxDepth=30, seed=33)



predictions = model.predict(transformed_df.map(lambda x: x.features))
model.save(sc,"s3://practicaltraining/trainingmodel")

predict_df = predictions.zipWithIndex().toDF()

result = predict_df.select((predict_df._2.cast('int').alias("id")), predict_df._1.alias("target")).cache()

result.show()


labels_and_predictions = transformed_df.map(lambda x: x.label).zip(predictions)

acc = labels_and_predictions.filter(lambda x: x[0] == 
x[1]).count() / float(transformed_df.count())

print("Model accuracy: %.3f%%" % (acc * 100))

metrics = MulticlassMetrics(labels_and_predictions)

fscore = metrics.fMeasure()
print("F1 Score = %s" % fscore)


