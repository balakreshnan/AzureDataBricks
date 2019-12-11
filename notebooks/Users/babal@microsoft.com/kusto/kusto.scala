// Databricks notebook source
//val KustoSparkTestAppId = dbutils.secrets.get(scope = "KustoDemos", key = "KustoSparkTestAppId")
//val KustoSparkTestAppKey = dbutils.secrets.get(scope = "KustoDemos", key = "KustoSparkTestAppKey")

val appId = "da4ee700-880e-4026-b700-5d77491cdfb5"
val appKey = "iN-?TOI=nawG1ml3U7*WA3I?KY2Lshrj"
val authorityId = "72f988bf-86f1-41af-91ab-2d7cd011db47"
val cluster = "adxbb.eastus2"
val database = "jemtsdb"
val table = "tsraw"

// COMMAND ----------

//non working mltrain app id
//val appId = "47761a08-eb6d-430b-b323-326d2165e17a"
//val appKey = "nu5EYqfgDVrDTPY0?7LQB1Gp.Qpx?ft+"
//val authorityId = "72f988bf-86f1-41af-91ab-2d7cd011db47"
//val cluster = "adxbb.eastus2"
//val database = "jemtsdb"
//val table = "tsraw"

// COMMAND ----------

import java.util.concurrent.atomic.AtomicInteger
import com.microsoft.kusto.spark.datasink
import com.microsoft.kusto.spark.datasource
import com.microsoft.kusto.spark.common
import com.microsoft.kusto.spark.utils.{KustoDataSourceUtils => KDSU}
import com.microsoft.kusto.spark.datasource.KustoSourceOptions
//import com.microsoft.azure.kusto.spark.datasink.KustoSinkOptions
//import com.microsoft.azure.kusto.spark.datasource.KustoSourceOptions
//import com.microsoft.azure.kusto.spark.utils.{KustoDataSourceUtils => KDSU}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructType}



// COMMAND ----------

// Simplified syntax flavor
import org.apache.spark.sql._
import com.microsoft.kusto.spark.sql.extension.SparkExtension._
import org.apache.spark.SparkConf

val conf: Map[String, String] = Map(
      KustoSourceOptions.KUSTO_AAD_CLIENT_ID -> appId,
      KustoSourceOptions.KUSTO_AAD_CLIENT_PASSWORD -> appKey,
      KustoSourceOptions.KUSTO_QUERY -> s"$table | limit 1000"      
    )



val df = spark.read.kusto(cluster, database, "", conf)
display(df)

// COMMAND ----------

// Simplified syntax flavor
import org.apache.spark.sql._
import com.microsoft.kusto.spark.sql.extension.SparkExtension._
import org.apache.spark.SparkConf

val conf: Map[String, String] = Map(
      KustoSourceOptions.KUSTO_AAD_CLIENT_ID -> appId,
      KustoSourceOptions.KUSTO_AAD_CLIENT_PASSWORD -> appKey,
      KustoSourceOptions.KUSTO_QUERY -> s"tsquarter | limit 10"      
    )



val dfquarterly = spark.read.kusto(cluster, database, "", conf)
display(dfquarterly)