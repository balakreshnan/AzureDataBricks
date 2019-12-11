// Databricks notebook source
dbutils.fs.ls("/mnt/data")

// COMMAND ----------

val dfoptel = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/data/Nozzle Performance with Optel 1020 GC3.csv")

// COMMAND ----------

dfoptel.printSchema()

// COMMAND ----------

dfoptel.dtypes

// COMMAND ----------

dfoptel.distinct.count

// COMMAND ----------

display(dfoptel.orderBy(asc("date_time")))

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import spark.implicits._
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.{concat, lit}


// COMMAND ----------

val df1=dfoptel.withColumn("date_time_x",unix_timestamp($"date_time", "MM/dd/yy HH:mm:ss").cast(TimestampType).as("timestamp"))

// COMMAND ----------



// COMMAND ----------

df1.printSchema

// COMMAND ----------

val df1withnozzlekey = df1.withColumn("nozzleKey", concat($"optel_schedule_wo", lit("_"), $"Ch-Hole"))

// COMMAND ----------

display(df1withnozzlekey)

// COMMAND ----------

df1withnozzlekey.dtypes

// COMMAND ----------

val df1asc = display(df1.orderBy(asc("date_time_x")))

// COMMAND ----------

import scala.math.{abs, max}
import scala.collection.mutable.ListBuffer

object NozzleSlopeTracker {
  
      class NozzleSlopeClass (maxSlopeCount : Int) {
        
        var _nzMap= scala.collection.mutable.Map.empty[String, String]
        var _maxSlopeCount : Integer = maxSlopeCount
           
        var _nzGeneratedId : String = ""
        var _nrSum1 : Double = 0
        var _nrSum2 : Double = 0
        var _nrSumPerc1 : Double = 0
        var _nrSumPerc2 : Double = 0
        var _slope : Double = 0
        var _nzdirection : String = "nil"
        var _outlier : Boolean = false
                
        def setSlope (nzGeneratedId : String, newNrSum : Double, newNrSumPerc : Double) {
          
          _nzGeneratedId = nzGeneratedId
          
        
          
          if(_nzMap.contains(nzGeneratedId))
          {
            
              val value = _nzMap(nzGeneratedId)
              println("Key: " +  nzGeneratedId + " exists with value :" + value)

               val blist = value.split(":")
            
               _nrSum1 = blist(0).toDouble
               _nrSum2 = blist(1).toDouble
               _nrSumPerc1 = blist(2).toDouble
               _nrSumPerc2 = blist(3).toDouble
               _slope = blist(4).toDouble
               _nzdirection = blist(5).toString
               _outlier = blist(6).toBoolean
            
              _nrSum1 = _nrSum2
              _nrSum2 = newNrSum
              _nrSumPerc1 = _nrSumPerc2
              _nrSumPerc2 = newNrSumPerc
            
              val lBuf = slope ()
            
              println("old buf: " + lBuf)
            
              _nzMap (nzGeneratedId) = lBuf
          }
          else
          {
              //println("$nzGeneratedId key does not exist")
            
               _nrSum1 = 0
               _nrSum2 = newNrSum
               _nrSumPerc1 = 0
               _nrSumPerc2 = newNrSumPerc
               _slope = 0
               _nzdirection = "nil"
               _outlier = false
            
               var lBuf = ""
               lBuf = lBuf.concat(_nrSum1.toString)
               lBuf = lBuf.concat(":")
               lBuf = lBuf.concat(_nrSum2.toString)
               lBuf = lBuf.concat(":")
               lBuf = lBuf.concat(_nrSumPerc1.toString)
               lBuf = lBuf.concat(":")
               lBuf = lBuf.concat(_nrSumPerc2.toString)
               lBuf = lBuf.concat(":")
               lBuf = lBuf.concat(_slope.toString)
               lBuf = lBuf.concat(":")
               lBuf = lBuf.concat(_nzdirection.toString)
               lBuf = lBuf.concat(":")
               lBuf = lBuf.concat(_outlier.toString)
               _nzMap(nzGeneratedId)=lBuf
            
            
              println(lBuf)
            
                                  
          }
        }
        
        def slope () : String = {
               
          if ((_nrSumPerc1 < _nrSumPerc2) && (_nrSum2 > _nrSum1)) {
               _nzdirection = "ru"
               _slope += 1
          }
          else if ((_nrSumPerc1 > _nrSumPerc2) && (_nrSum2 > _nrSum1)) {
               _nzdirection= "lu"
               _slope += 1
          }
          else if ((_nrSumPerc1 < _nrSumPerc2) && (_nrSum2 == _nrSum1)) {
               if (_nzdirection == "l") {
                   _slope = 0
               }
               else {             
                   _slope += 1
               }
               _nzdirection= "r"
          }
          else if ((_nrSumPerc1 > _nrSumPerc2) && (_nrSum2 == _nrSum1)) {
            
               if (_nzdirection == "r") {
                   _slope = 0
               }
               else {             
                   _slope += 1
               }
               _nzdirection= "l"
          }
          
          _outlier = false
          
          if (_slope >= _maxSlopeCount) {
            _outlier = true
            println("nozzle key: " + _nzGeneratedId + " is outlier -> TRUE")
          }
          
          var lBuf = ""
          lBuf = lBuf.concat(_nrSum1.toString)
          lBuf = lBuf.concat(":")
          lBuf = lBuf.concat(_nrSum2.toString)
          lBuf = lBuf.concat(":")
          lBuf = lBuf.concat(_nrSumPerc1.toString)
          lBuf = lBuf.concat(":")
          lBuf = lBuf.concat(_nrSumPerc2.toString)
          lBuf = lBuf.concat(":")
          lBuf = lBuf.concat(_slope.toString)
          lBuf = lBuf.concat(":")
          lBuf = lBuf.concat(_nzdirection)
          lBuf = lBuf.concat(":")
          lBuf = lBuf.concat(_outlier.toString)
          
          println("new nozzle vector: " + lBuf)
         
          lBuf
        }
        
        def reset (nrSum : Double, nrSumPerc : Double) {
           /* We do not know when and how to reset yet.*/
        }
        
        def getMe () : NozzleSlopeClass = {
          this
        }
    }
}

// COMMAND ----------

df1asc

// COMMAND ----------



val nozzles = new NozzleSlopeTracker.NozzleSlopeClass(3)


//Bala, tell me how to pass each the 'Noz Rejects Sum%'	'Noz Rejects Sum' 'nozzleKey' from dataframe : df1withnozzlekey, into nozzles instance

//nozzles.setSlope('nozzleKey', 'Noz Rejects Sum%', 'Noz Rejects Sum')

