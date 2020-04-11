// Databricks notebook source
// MAGIC %md ##On-Time Flight Performance with GraphFrames for Apache Spark
// MAGIC This notebook provides an analysis of On-Time Flight Performance and Departure Delays data using GraphFrames for Apache Spark.
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC Source Data:  
// MAGIC * [OpenFlights: Airport, airline and route data](http://openflights.org/data.html)
// MAGIC * [United States Department of Transportation: Bureau of Transportation Statistics (TranStats)](http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time)

// COMMAND ----------

// MAGIC %md
// MAGIC ### **Step 1:** Data Ingestion
// MAGIC Extract the Airports and Departure Delays information from DBFS

// COMMAND ----------

val airportsnaFilePath = "/databricks-datasets/flights/airport-codes-na.txt"
val tripdelaysFilePath = "/databricks-datasets/flights/departuredelays.csv"

// COMMAND ----------

// MAGIC %md Create Airports Dataset

// COMMAND ----------

//Obtain airports dataset
val airportsna = spark
                  .read
                  .format("csv")
                  .options(Map("header" -> "true", 
                               "inferschema" -> "true", 
                               "delimiter" -> "\t"))
                  .load(airportsnaFilePath)

airportsna.createOrReplaceTempView("airports_na")

// COMMAND ----------

// MAGIC %md Create Departure Delays Dataset

// COMMAND ----------

val departureDelays = spark
                        .read
                        .format("csv")
                        .option("header", "true")
                        .load(tripdelaysFilePath)

departureDelays.createOrReplaceTempView("departureDelays")

// COMMAND ----------

// MAGIC %md
// MAGIC ### **Step 2:** Data Preparation

// COMMAND ----------

// MAGIC %md Extract IATA codes from the departuredelays sample dataset

// COMMAND ----------

// MAGIC %sql 
// MAGIC CREATE OR REPLACE TEMP VIEW tripIATA AS 
// MAGIC   SELECT DISTINCT iata FROM 
// MAGIC     (SELECT DISTINCT origin AS iata FROM departureDelays 
// MAGIC      UNION ALL 
// MAGIC      SELECT DISTINCT destination AS iata FROM departureDelays) a

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMP VIEW airports AS
// MAGIC   SELECT f.IATA, f.City, f.State, f.Country 
// MAGIC   FROM airports_na f JOIN tripIATA t ON t.IATA = f.IATA

// COMMAND ----------

// MAGIC %md Obtain key attributes such as Date of flight, delays, distance, and airport information (Origin, Destination)

// COMMAND ----------

// MAGIC %sql 
// MAGIC CREATE OR REPLACE TEMP VIEW departureDelays_geo AS
// MAGIC   SELECT CAST(f.date AS int) tripid, 
// MAGIC   CAST(concat(concat(concat(concat(concat(concat('2014-', concat(concat(substr(CAST(f.date AS string), 1, 2), '-')), 
// MAGIC   substr(CAST(f.date AS string), 3, 2)), ' '), 
// MAGIC   substr(CAST(f.date AS string), 5, 2)), ':'), 
// MAGIC   substr(CAST(f.date AS string), 7, 2)), ':00') AS timestamp) `localdate`, 
// MAGIC   CAST(f.delay AS int), 
// MAGIC   CAST(f.distance AS int), 
// MAGIC   f.origin src, 
// MAGIC   f.destination dst, 
// MAGIC   o.city city_src, 
// MAGIC   d.city city_dst, 
// MAGIC   o.state state_src, 
// MAGIC   d.state state_dst 
// MAGIC   FROM departuredelays f JOIN airports o ON o.iata = f.origin 
// MAGIC   JOIN airports d ON d.iata = f.destination

// COMMAND ----------

// MAGIC %sql select * from departuredelays limit 100

// COMMAND ----------

// MAGIC %sql select * from departureDelays_geo limit 100

// COMMAND ----------

// MAGIC %md
// MAGIC ## Building the Graph
// MAGIC Now that we've imported our data, we're going to need to build our graph. To do so we're going to do two things. We are going to build the structure of the vertices (or nodes) and we're going to build the structure of the edges. What's awesome about GraphFrames is that this process is incredibly simple. 
// MAGIC * Rename IATA airport code to **id** in the Vertices Table
// MAGIC * Start and End airports to **src** and **dst** for the Edges Table (flights)

// COMMAND ----------

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import org.graphframes._

// COMMAND ----------

// MAGIC %md Define the Vertices and Edges of the Graph

// COMMAND ----------

val tripVertices = sql("SELECT IATA id, City, State, Country FROM airports")
val tripEdges = sql("SELECT tripid, delay, src, dst, city_dst, state_dst FROM departureDelays_geo")

// COMMAND ----------

//display graph vertices

display(tripVertices)

// COMMAND ----------

//display graph edges

display(tripEdges)

// COMMAND ----------

// MAGIC %md ####Build the Graph

// COMMAND ----------

val tripGraph = GraphFrame(tripVertices, tripEdges)

//Build `tripGraphPrime` GraphFrame
//This graphframe contains a smaller subset of data to make it easier to display motifs and subgraphs (below)
val tripEdgesPrime = sql("SELECT tripid, delay, src, dst FROM departureDelays_geo")
val tripGraphPrime = GraphFrame(tripVertices, tripEdgesPrime)

// COMMAND ----------

// MAGIC %md #Step 3: Data Analysis

// COMMAND ----------

// MAGIC %md #### Determine the number of airports and trips

// COMMAND ----------

println(s"Airports: ${tripGraph.vertices.count()}")
println(s"Trips: ${tripGraph.edges.count()}")

// COMMAND ----------

// MAGIC %md ## Visualize airports and flight paths using D3 Visuals

// COMMAND ----------

package d3a
// We use a package object so that we can define top level classes like Edge that need to be used in other cells

import org.apache.spark.sql._
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML

case class Edge(src: String, dest: String, count: Long)

case class Node(name: String)
case class Link(source: Int, target: Int, value: Long)
case class Graph(nodes: Seq[Node], links: Seq[Link])

object graphs {
val sqlContext = SparkSession.builder().getOrCreate()
import sqlContext.implicits._

def force(clicks: Dataset[Edge], height: Int = 100, width: Int = 960): Unit = {
  val data = clicks.collect()
  val nodes = (data.map(_.src) ++ data.map(_.dest)).map(_.replaceAll("_", " ")).toSet.toSeq.map(Node)
  val links = data.map { t =>
    Link(nodes.indexWhere(_.name == t.src.replaceAll("_", " ")), nodes.indexWhere(_.name == t.dest.replaceAll("_", " ")), t.count / 20 + 1)
  }
  showGraph(height, width, Seq(Graph(nodes, links)).toDF().toJSON.first())
}

/**
 * Displays a force directed graph using d3
 * input: {"nodes": [{"name": "..."}], "links": [{"source": 1, "target": 2, "value": 0}]}
 */
def showGraph(height: Int, width: Int, graph: String): Unit = {

displayHTML(s"""<!DOCTYPE html>
<html>
  <head>
    <link type="text/css" rel="stylesheet" href="https://mbostock.github.io/d3/talk/20111116/style.css"/>
    <style type="text/css">
      #states path {
        fill: #ccc;
        stroke: #fff;
      }

      path.arc {
        pointer-events: none;
        fill: none;
        stroke: #000;
        display: none;
      }

      path.cell {
        fill: none;
        pointer-events: all;
      }

      circle {
        fill: steelblue;
        fill-opacity: .8;
        stroke: #fff;
      }

      #cells.voronoi path.cell {
        stroke: brown;
      }

      #cells g:hover path.arc {
        display: inherit;
      }
    </style>
  </head>
  <body>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.js"></script>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.csv.js"></script>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geo.js"></script>
    <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geom.js"></script>
    <script>
      var graph = $graph;
      var w = $width;
      var h = $height;

      var linksByOrigin = {};
      var countByAirport = {};
      var locationByAirport = {};
      var positions = [];

      var projection = d3.geo.azimuthal()
          .mode("equidistant")
          .origin([-98, 38])
          .scale(1400)
          .translate([640, 360]);

      var path = d3.geo.path()
          .projection(projection);

      var svg = d3.select("body")
          .insert("svg:svg", "h2")
          .attr("width", w)
          .attr("height", h);

      var states = svg.append("svg:g")
          .attr("id", "states");

      var circles = svg.append("svg:g")
          .attr("id", "circles");

      var cells = svg.append("svg:g")
          .attr("id", "cells");

      var arc = d3.geo.greatArc()
          .source(function(d) { return locationByAirport[d.source]; })
          .target(function(d) { return locationByAirport[d.target]; });

      d3.select("input[type=checkbox]").on("change", function() {
        cells.classed("voronoi", this.checked);
      });

      // Draw US map.
      d3.json("https://mbostock.github.io/d3/talk/20111116/us-states.json", function(collection) {
        states.selectAll("path")
          .data(collection.features)
          .enter().append("svg:path")
          .attr("d", path);
      });

      // Parse links
      graph.links.forEach(function(link) {
        var origin = graph.nodes[link.source].name;
        var destination = graph.nodes[link.target].name;

        var links = linksByOrigin[origin] || (linksByOrigin[origin] = []);
        links.push({ source: origin, target: destination });

        countByAirport[origin] = (countByAirport[origin] || 0) + 1;
        countByAirport[destination] = (countByAirport[destination] || 0) + 1;
      });

      d3.csv("https://mbostock.github.io/d3/talk/20111116/airports.csv", function(data) {

        // Build list of airports.
        var airports = graph.nodes.map(function(node) {
          return data.find(function(airport) {
            if (airport.iata === node.name) {
              var location = [+airport.longitude, +airport.latitude];
              locationByAirport[airport.iata] = location;
              positions.push(projection(location));

              return true;
            } else {
              return false;
            }
          });
        });

        // Compute the Voronoi diagram of airports' projected positions.
        var polygons = d3.geom.voronoi(positions);

        var g = cells.selectAll("g")
            .data(airports)
          .enter().append("svg:g");

        g.append("svg:path")
            .attr("class", "cell")
            .attr("d", function(d, i) { return "M" + polygons[i].join("L") + "Z"; })
            .on("mouseover", function(d, i) { d3.select("h2 span").text(d.name); });

        g.selectAll("path.arc")
            .data(function(d) { return linksByOrigin[d.iata] || []; })
          .enter().append("svg:path")
            .attr("class", "arc")
            .attr("d", function(d) { return path(arc(d)); });

        circles.selectAll("circle")
            .data(airports)
            .enter().append("svg:circle")
            .attr("cx", function(d, i) { return positions[i][0]; })
            .attr("cy", function(d, i) { return positions[i][1]; })
            .attr("r", function(d, i) { return Math.sqrt(countByAirport[d.iata]); })
            .sort(function(a, b) { return countByAirport[b.iata] - countByAirport[a.iata]; });
      });
    </script>
  </body>
</html>""")
  }

  def help() = {
displayHTML("""
<p>
Produces a force-directed graph given a collection of edges of the following form:</br>
<tt><font color="#a71d5d">case class</font> <font color="#795da3">Edge</font>(<font color="#ed6a43">src</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">dest</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">count</font>: <font color="#a71d5d">Long</font>)</tt>
</p>
<p>Usage:<br/>
<tt>%scala</tt></br>
<tt><font color="#a71d5d">import</font> <font color="#ed6a43">d3._</font></tt><br/>
<tt><font color="#795da3">graphs.force</font>(</br>
&nbsp;&nbsp;<font color="#ed6a43">height</font> = <font color="#795da3">500</font>,<br/>
&nbsp;&nbsp;<font color="#ed6a43">width</font> = <font color="#795da3">500</font>,<br/>
&nbsp;&nbsp;<font color="#ed6a43">clicks</font>: <font color="#795da3">Dataset</font>[<font color="#795da3">Edge</font>])</tt>
</p>""")
  }
}

// COMMAND ----------

import d3a._
graphs.force(
  height = 800,
  width = 1200,
  clicks = sql("""select src, dst as dest, count(*) as count from departureDelays_geo where state_src in ('CA', 'OR', 'WA') and delay > 0 group by src, dst""").as[Edge])

// COMMAND ----------

// MAGIC %md #### Determining the number of delayed vs. on-time / early flights

// COMMAND ----------

println(s"On-time / Early Flights: ${tripGraph.edges.filter("delay <= 0").count()}")
println(s"Delayed Flights: ${tripGraph.edges.filter("delay > 0").count()}")

// COMMAND ----------

// MAGIC %md #### What flights departing SFO are most likely to have significant delays
// MAGIC Note, delay can be <= 0 meaning the flight left on time or early

// COMMAND ----------

tripGraph.edges
  .filter("src = 'SFO' and delay > 0")
  .groupBy("src", "dst")
  .avg("delay")
  .sort(desc("avg(delay)"))

// COMMAND ----------

display(tripGraph.edges.filter("src = 'SFO' and delay > 0").groupBy("src", "dst").avg("delay").sort(desc("avg(delay)")))

// COMMAND ----------

// MAGIC %md #### Which destinations tend to have significant delays departing from SEA

// COMMAND ----------

//States with the longest cumulative delays (with individual delays > 100 minutes) (origin: Seattle)
display(tripGraph.edges.filter("src = 'SEA' and delay > 100"))

// COMMAND ----------

// MAGIC %md ## Finding motifs
// MAGIC What delays are occuring in SFO?

// COMMAND ----------

val motifs = tripGraph.find("(a)-[e1]->(b); (b)-[e2]->(c)").filter("(b.id = 'SFO') and (e1.delay > 500 and e2.delay > 500) and (e1.tripid < e2.tripid)");
display(motifs)

// COMMAND ----------

// MAGIC %md ## Determining Airport Ranking by connections using PageRank
// MAGIC There are a large number of flights and connections through these various airports included in this Departure Delay Dataset.  Using the `pageRank` algorithm, Spark iteratively traverses the graph and determines a rough estimate of how important the airport is.

// COMMAND ----------

//Determining Airport ranking of importance using `pageRank`
val ranks = tripGraph.pageRank.resetProbability(0.15).maxIter(5).run()
val verticesDf = ranks.vertices
display(verticesDf.orderBy($"pagerank".desc).limit(20))

// COMMAND ----------

// MAGIC %md ## Most popular flights (single city hops)
// MAGIC Using the `tripGraph`, we can quickly determine what are the most popular single city hop flights

// COMMAND ----------

//Determine the most popular flights (single city hops)
val topTrips = tripGraph
                .edges
                .groupBy("src", "dst")
                .agg(count("delay").alias("trips"))

// COMMAND ----------

//Show the top 20 most popular flights (single city hops)
display(topTrips.orderBy($"trips".desc).limit(20))