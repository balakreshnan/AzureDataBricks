// Databricks notebook source
dbutils.fs.ls("/mnt/data")

// COMMAND ----------

val dfoptel = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/data/NozzlewithOptel1020GC3.csv")

// COMMAND ----------

dfoptel.printSchema()

// COMMAND ----------

dfoptel.dtypes

// COMMAND ----------

dfoptel.distinct.count

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import spark.implicits._
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.{concat, lit}


// COMMAND ----------

display(dfoptel.orderBy(asc("date_time"),asc("Ch-Hole")))

// COMMAND ----------

display(dfoptel.orderBy(asc("Ch-Hole"),asc("date_time")))

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

val df1asc = df1withnozzlekey.orderBy(asc("date_time_x"))

// COMMAND ----------

display(df1asc)

// COMMAND ----------

val jdbcUsername = dbutils.secrets.get(scope = "jdbc", key = "username")
val jdbcPassword = dbutils.secrets.get(scope = "jdbc", key = "password")

// COMMAND ----------

Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver")

// COMMAND ----------



val jdbcHostname = "dewdbsvr.database.windows.net"
val jdbcPort = 1433
val jdbcDatabase ="dewdb"

// Create the JDBC URL without passing in the user and password parameters.
val jdbcUrl = s"jdbc:sqlserver://${jdbcHostname}:${jdbcPort};database=${jdbcDatabase}"

// Create a Properties() object to hold the parameters.
import java.util.Properties
val connectionProperties = new Properties()
connectionProperties.put("user", s"sqladmin")
connectionProperties.put("password", s"Azure!2345")


// COMMAND ----------

val driverClass = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
connectionProperties.setProperty("Driver", driverClass)

// COMMAND ----------

val dfnozzlestatus = spark.read.jdbc(jdbcUrl, "nozzlestatus", connectionProperties)
val dfnozzlestatusrow = spark.read.jdbc(jdbcUrl, "nozzlestatus", connectionProperties)

// COMMAND ----------

display(dfnozzlestatus)

// COMMAND ----------

case class nozzlestatus(timehappened: String, nozzleid: String, status: String, plant: String, machine: String)
case class nozzlelist(nozzlestatus: Seq[nozzlestatus])

// COMMAND ----------

val nozzlestatus1 = new nozzlestatus("123456", "MachineID", "TRUE")
val nzlist = new nozzlelist(Seq(nozzlestatus1))

// COMMAND ----------


import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

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
        var _thappened : String = ""
                
        def setSlope (nzGeneratedId : String, newNrSum : Double, newNrSumPerc : Double, thappened: String) {
          
          _nzGeneratedId = nzGeneratedId
          _thappened = thappened
          
        
          
          if(_nzMap.contains(nzGeneratedId))
          {
            
              val value = _nzMap(nzGeneratedId)
              //println("Key: " +  nzGeneratedId + " exists with value :" + value)

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
            
              //println("old buf: " + lBuf)
            
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
            
            
              //println(lBuf)
            
                                  
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
            //println("nozzle key: " + _nzGeneratedId + " is outlier -> TRUE")
            println(_thappened + "," + _nzGeneratedId + ",TRUE" + "," + _nzdirection + "," + _slope)
            //val nozzlestatus1 = new nozzlestatus("06/01/2018", _nzGeneratedId, "TRUE", "TWB","GC10")
            //val nozzlestatus1 = new nozzlestatus("06/01/2018", _nzGeneratedId, "TRUE", "TWB","GC10")
            val nozzlestatus1 = new nozzlestatus(_thappened, _nzGeneratedId, "TRUE", "TWB","GC10")
            //val df1 = nozzlestatus1.toDF()
            //val departmentsWithEmployeesSeq1 = Seq(nozzlestatus1)
            //val df1 = departmentsWithEmployeesSeq1.toDF()
            //display(df1)
            //dfnozzlestatus.union(df1)
            //nzlist = new nozzlelist(Seq(nozzlestatus1))
            //val df2 = df.withColumn("machineid", lit(_nzGeneratedId)).withColumn("status", lit("TRUE"))
            
            val dfwrite = Seq(nozzlestatus1).toDF()
            
            //dfwrite.write.mode("append").jdbc(jdbcUrl, "nozzlestatus", connectionProperties)
            //dfwrite.write.mode("overwrite").jdbc(jdbcUrl, "nozzlestatus", connectionProperties)
            //dfwrite.write.mode("ignore").jdbc(jdbcUrl, "nozzlestatus", connectionProperties)
            dfwrite.write.mode("append").jdbc(jdbcUrl, "nozzlestatus", connectionProperties)
            

            //df.union(df2)
            
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
          
          //println("new nozzle vector: " + lBuf)
         
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

display(df1asc)

// COMMAND ----------



val nozzles = new NozzleSlopeTracker.NozzleSlopeClass(10)


//Bala, tell me how to pass each the 'Noz Rejects Sum%'	'Noz Rejects Sum' 'nozzleKey' from dataframe : df1withnozzlekey, into nozzles instance

//nozzles.setSlope('nozzleKey', 'Noz Rejects Sum%', 'Noz Rejects Sum')



// COMMAND ----------

df1asc.printSchema

// COMMAND ----------

def calc(t: Row)
{
  var _nrSum1 : Double = 0
  var _nrSum2 : Double = 0
  var _nrKey : String = ""
  var _thappened : String = ""
  _nrSum1 = t.getAs[Double]("Noz Rejects Sum%")
  _nrSum2 = t.getAs[Int]("Noz Rejects Sum")
  _nrKey = t.getAs[String]("nozzleKey")
  _thappened = t.getAs[String]("date_time")
  
  
  nozzles.setSlope(_nrKey, _nrSum1, _nrSum2,_thappened)
  //println(t.getAs[String]("date_time"),t.getAs[String]("nozzleKey"),t.getAs[String]("Noz Rejects Sum%"),t.getAs[String]("Noz Rejects Sum"))
}

// COMMAND ----------

df1asc.collect().foreach(t => calc(t))
//nozzles._nzmap = scala.collection.mutable.Map.empty[String, String]
//nozzles._nzmap.clear

// COMMAND ----------

display(df1asc)

// COMMAND ----------

val df1rs = df1asc.foreach(t => calc(t))

// COMMAND ----------

display(df1rs)

// COMMAND ----------

val jdbcHostname = "dewdbsvr.database.windows.net:1433" 
val jdbcPort = 1433 
val jdbcDatabase ="dewdb" 
// Create the JDBC URL without passing in the user and password parameters. 
val jdbcUrl = s"jdbc:sqlserver://dewdbsvr.database.windows.net:1433;database=dewdb" 
// Create a Properties() object to hold the parameters. 
import java.util.Properties 
val connectionProperties = new Properties() 
connectionProperties.put("user", s"sqladmin") 
connectionProperties.put("password", s"Azure!2345")

val TRIPS_BY_HOUR = spark.read.jdbc(jdbcUrl, "TRIPS_BY_HOUR", connectionProperties)
display(TRIPS_BY_HOUR)

// COMMAND ----------



// COMMAND ----------

display(df1asc.where(df1asc("Noz Rejects Sum") > 0))

// COMMAND ----------

display(df1asc.orderBy(asc("optel_schedule_wo"),asc("Ch-Hole"),asc("date_time")).where(df1asc("Noz Rejects Sum") > 0))

// COMMAND ----------

val df1fil = df1asc.orderBy(asc("optel_schedule_wo"),asc("Ch-Hole"),asc("date_time")).where(df1asc("Noz Rejects Sum") > 0)

// COMMAND ----------

display(df1fil)

// COMMAND ----------

import org.apache.spark.sql.expressions.Window

// COMMAND ----------

//val prev : Row


def calc1(t: Row)
{
  var _nrSum1 : Double = 0
  var _nrSum2 : Double = 0
  var _nrKey : String = ""
  _nrSum1 = t.getAs[Double]("Noz Rejects Sum%")
  _nrSum2 = t.getAs[Int]("Noz Rejects Sum")
  _nrKey = t.getAs[String]("nozzleKey")
  
  
  //nozzles.setSlope(_nrKey, _nrSum1, _nrSum2)
  println(t.getAs[String]("date_time"),t.getAs[String]("nozzleKey"),t.getAs[String]("Noz Rejects Sum%"),t.getAs[String]("Noz Rejects Sum"))
}

// COMMAND ----------

df1fil.collect().foreach(t => calc1(t))

// COMMAND ----------

val w = Window.orderBy(asc("optel_schedule_wo"),asc("Ch-Hole"),asc("date_time"))
df1fil.withColumn("Noz Rejects Sum%", 
    when($"Ch-Hole" === lag($"Ch-Hole", 1).over(w), lag($"Noz Rejects Sum%", 1).over(w)).otherwise($"Noz Rejects Sum%")
).show

// COMMAND ----------

val w = Window.partitionBy("Ch-Hole").orderBy(asc("optel_schedule_wo"),asc("Ch-Hole"),asc("date_time"))
df1fil.withColumn("Noz Rejects Sum", 
    when($"Ch-Hole" === lag($"Ch-Hole", 1).over(w), lag($"Noz Rejects Sum", 1).over(w)).otherwise($"Noz Rejects Sum")
).show

// COMMAND ----------

display(df1fil.where(df1fil("Noz Rejects Sum") > 3))

// COMMAND ----------

val w = Window.partitionBy("Ch-Hole").orderBy(asc("optel_schedule_wo"),asc("Ch-Hole"),asc("date_time"))
df1fil.withColumn("Noz Rejects Sum", 
    when($"Ch-Hole" === lag($"Ch-Hole", 1).over(w), lag($"Noz Rejects Sum", 1).over(w)).otherwise($"Noz Rejects Sum")
).show

// COMMAND ----------

val windowedCounts = df1fil.groupBy(window(df1fil.col("date_time_x"), "75 minutes", "75 minutes"),$"nozzleKey"
).count()

// COMMAND ----------

display(windowedCounts.orderBy(asc("window"),asc("nozzleKey")))

// COMMAND ----------

val q = Window.partitionBy("Ch-Hole").orderBy(asc("optel_schedule_wo"),asc("Ch-Hole"),asc("date_time")).rowsBetween(-4, 0)

// COMMAND ----------

val ranked = df1fil.withColumn("rank", dense_rank.over(q))

// COMMAND ----------

display(ranked)