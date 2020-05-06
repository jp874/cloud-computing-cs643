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
from pyspark.mllib.tree import RandomForest,RandomForestModel
sc = SparkContext()
spark = SparkSession(sc)

inputDF = spark.read.csv('s3://practicaltraining/V.csv',header='true', inferSchema='true', sep=';')

#transformed_df= inputDF.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

transformed_df= inputDF.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))




m=RandomForestModel.load(sc,"s3://practicaltraining/trainingmodel1")


predictions = m.predict(transformed_df.map(lambda x: x.features))

predict_df = predictions.zipWithIndex().toDF()

result = predict_df.select((predict_df._2.cast('int').alias("id")), predict_df._1.alias("target")).cache()

result.show()



labels_and_predictions = transformed_df.map(lambda x: x.label).zip(predictions)
predict_df1= labels_and_predictions.zipWithIndex().toDF()

result2 = predict_df1.select((predict_df1._2.cast('int').alias("id")), predict_df1._1.alias("target")).cache()

result2.show()

acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(transformed_df.count())
print("Model accuracy: %.3f%%" % (acc * 100))


metrics = MulticlassMetrics(labels_and_predictions)

fscore = metrics.fMeasure()
print("F1 Score = %s" % fscore)
