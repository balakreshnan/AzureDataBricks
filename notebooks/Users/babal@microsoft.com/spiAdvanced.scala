// Databricks notebook source
dbutils.fs.ls("/mnt/data")

// COMMAND ----------

val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/data/query-impala-70510.csv")

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


val aggdf1=df.groupBy($"generated_key", $"ref_designator").agg(avg($"solder_paste_volume") as ("solder_paste_average"))

// COMMAND ----------

val splitdf1=df.withColumn("generated_key",split(df("generated_key"),"-").getItem(0))

// COMMAND ----------

val aggdf2=splitdf1.groupBy($"generated_key").agg(count($"generated_key") as ("count_per_panel"))

// COMMAND ----------

val aggdf1=splitdf1.groupBy($"generated_key", $"ref_designator").agg(avg($"solder_paste_volume") as ("solder_paste_average"))

// COMMAND ----------

val aggdf1=splitdf1.groupBy($"generated_key", $"pin_nbr").agg(avg($"solder_paste_volume") as ("solder_paste_average"))

// COMMAND ----------

val aggdf2 = splitdf1.where(splitdf1("solder_paste_volume") <= 60.0)

// COMMAND ----------

aggdf2.count

// COMMAND ----------

case class refdesAgg(generated_key: String, pin_nbr: String, solder_paste_average: String)

// COMMAND ----------

object DataSetHeader {
  
      class NexSetClass (df : DataFrame) {
        
        var _df : DataFrame = df
        var _defaultStep : Integer = 1000
        var _from : Integer = 0
        
        val _schema : StructType = new StructType()
                                      .add(StructField("generated_key", StringType, true))
                                      .add(StructField("pin_nbr", StringType, true))
                                      .add(StructField("paste_volume_average", DoubleType, true))
        def next () { 
       
            val collection = _df.collect().slice(_from, _from + _defaultStep)
            _from = _from + _defaultStep

            val distDataRDD = sc.parallelize(collection)

            val schema = new StructType()
              .add(StructField("generated_key", StringType, true))
              .add(StructField("pin_nbr", StringType, true))
              .add(StructField("paste_volume_average", DoubleType, true))

            val df = spark.createDataFrame(distDataRDD, schema)
            display(df)
        }
        
        def back () { 
            
            _from = _from - _defaultStep
            if (_from <= 0) 
              _from = 0

            val collection = _df.collect().slice(_from, _from + _defaultStep)
           
            val distDataRDD = sc.parallelize(collection)

            val schema = new StructType()
              .add(StructField("generated_key", StringType, true))
              .add(StructField("pin_nbr", StringType, true))
              .add(StructField("paste_volume_average", DoubleType, true))

            val df = spark.createDataFrame(distDataRDD, schema)
            display(df)
        }
        
        def nextFrom (from : Integer) {    
            _from = from
            next()           
        }
        
        def nextFromBy (from : Integer, by : Integer) { 
          
            _from = from
            _defaultStep = by
            next()
        }
        def showLimits () { 
          
          println(_from)
          println(_defaultStep)
        }
   }
}

// COMMAND ----------

val dsh = new DataSetHeader.NexSetClass(aggdf1)

// COMMAND ----------

dsh.next()

// COMMAND ----------

dsh.nextFromBy(0,10)

// COMMAND ----------

dsh.nextFrom(0)

// COMMAND ----------

dsh.back()

// COMMAND ----------

dsh.showLimits()