# Databricks notebook source
# MAGIC %md # Combining Pre-Built & Custom AI Services

# COMMAND ----------

# MAGIC %md In this notebook, you will integrate with the Computer Vision API and the Text Analytics API to augment the claims processing capabilities. In the end, you will integrate the API calls to the summarizer and classifier services that your deployed and produce a finished claim report that shows all of the processing applied to the claim text and claim image.

# COMMAND ----------

# MAGIC %md ## Task 1 - Caption & Tag with the Computer Vision API

# COMMAND ----------

# MAGIC %md In the cell bellow, provided the key to your Computer Vision API and run the cell.

# COMMAND ----------

subscription_key = "d5fdb38303234276989074cab599a4b4"
assert subscription_key

# COMMAND ----------

# MAGIC %md Construct the endpoint to the Computer Vision API by running the following cell. Notice the last path segment is analyze, which indicates you will use the analyze feature.
# MAGIC 
# MAGIC Be sure to update the value in `vision_endpoint` below so it matches the Endpoint value you copied from the Azure Portal for your instance of the Computer Vision service. Be sure your value ends in a slash (/)

# COMMAND ----------

vision_endpoint = "https://westus2.api.cognitive.microsoft.com/"
vision_base_url = vision_endpoint + "vision/v1.0/"
vision_analyze_url = vision_base_url + "analyze"

# COMMAND ----------

# MAGIC %md The following cell contains a list of sample images found after performing a simple web search. Feel free to substitute in URL's to the image of your choice. 

# COMMAND ----------

fender_bender = "http://ford-life.com/wp-content/uploads/2012/03/Fender-Bender-image.jpg"
damaged_house = "https://c2.staticflickr.com/8/7342/10983313185_0589b74946_z.jpg"
police_car = "https://localtvwnep.files.wordpress.com/2015/11/fender-bender.jpeg?quality=85&strip=all"
car_with_text = "https://static.buildasign.com/cmsimages/bas-vinyl-lettering-splash-01.png?v=4D66584B51394A676D68773D"

# COMMAND ----------

# MAGIC %md From the list of images above, select one and assign it to image_url for further processing:

# COMMAND ----------

image_url = police_car

# COMMAND ----------

# MAGIC %md Run the following cell to preview the image you have selected.

# COMMAND ----------

displayHTML(image_url)

# COMMAND ----------

# MAGIC %md The following cell builds the HTTP request to make against the Computer Vision API. 
# MAGIC 
# MAGIC Run the following cell to retrieve the caption and tags:

# COMMAND ----------

import requests
headers  = {'Ocp-Apim-Subscription-Key': subscription_key }
params   = {'visualFeatures': 'Categories,Description,Tags,Color'}
data     = {'url': image_url}
response = requests.post(vision_analyze_url, headers=headers, params=params, json=data)
response.raise_for_status()
analysis = response.json()
analysis

# COMMAND ----------

# MAGIC %md As you can see in the above output, the result is a nested document structure. Run the following cells to pull out the caption and top 3 tag results:

# COMMAND ----------

caption = analysis["description"]["captions"][0]["text"].capitalize()
caption

# COMMAND ----------

topTags = analysis["description"]["tags"][0:3]
topTags

# COMMAND ----------

# MAGIC %md ## Task 2 - Performing OCR

# COMMAND ----------

# MAGIC %md In order to perform OCR with the Computer Vision service, you need to target the OCR endpoint. 
# MAGIC 
# MAGIC Run the following cell to construct the right URL:

# COMMAND ----------

vision_ocr_url = vision_base_url + "ocr"

# COMMAND ----------

# MAGIC %md Next, invoke the OCR endpoint with the following code and examine the result:

# COMMAND ----------

headers  = {'Ocp-Apim-Subscription-Key': subscription_key }
params   = {}
data     = {'url': image_url}
response = requests.post(vision_ocr_url, headers=headers, params=params, json=data)
response.raise_for_status()
ocr_analysis = response.json()
ocr_analysis

# COMMAND ----------

# MAGIC %md We have provided the following code for you to extract the text as a flat array from the results.
# MAGIC 
# MAGIC Run the following cell to extract the text items from the results document:

# COMMAND ----------

import itertools
flatten = lambda x: list(itertools.chain.from_iterable(x))
words_list = [[ [w['text'] for w in line['words']]  for line in d['lines']] for d in ocr_analysis['regions']]
words_list = flatten(flatten(words_list))
print(list(words_list))

# COMMAND ----------

# MAGIC %md ## Task 3 - Performing Sentiment Analysis

# COMMAND ----------

# MAGIC %md Sentiment Analysis is performed using the Text Analytics API. 
# MAGIC 
# MAGIC Update the following cell with the key to your instance of the Text Analytics API and run the cell:

# COMMAND ----------

text_analytics_subscription_key = "84b772a3a73b41718ac6f05816220f72"
assert text_analytics_subscription_key

# COMMAND ----------

# MAGIC %md Update the following cell with the correct base URL for your deployed instance of the Text Analytics API and run the cell:

# COMMAND ----------

text_analytics_base_url = "https://westus2.api.cognitive.microsoft.com/text/analytics/v2.0/"
sentiment_api_url = text_analytics_base_url + "sentiment"

# COMMAND ----------

# MAGIC %md The following cell has a set of example claims you can use to test the measurement sentiment. 
# MAGIC 
# MAGIC Run the cell:

# COMMAND ----------

neg_sent = """We are just devastated and emotionally drained. 
The roof was torn off of our car, and to make matters
worse my daughter's favorite teddy bear was impaled on the street lamp."""
pos_sent = """We are just happy the damaage was mininmal and that everyone is safe. 
We are thankful for your support."""
neutral_sent = """I crashed my car."""
long_claim = """
I was driving down El Camino and stopped at a red light.
It was about 3pm in the afternoon.  
The sun was bright and shining just behind the stoplight.
This made it hard to see the lights.
There was a car on my left in the left turn lane.
A few moments later another car, a black sedan pulled up behind me. 
When the left turn light changed green, the black sedan hit me thinking 
that the light had changed for us, but I had not moved because the light 
was still red.
After hitting my car, the black sedan backed up and then sped past me.
I did manage to catch its license plate. 
The license plate of the black sedan was ABC123. 
"""

# COMMAND ----------

# MAGIC %md From the above list of claims, select one and assign its variable to claim_text to be used in the call to the Text Analytics API.

# COMMAND ----------

claim_text = long_claim

# COMMAND ----------

# MAGIC %md The API requires you to submit a document of the following form. 
# MAGIC 
# MAGIC Run the cell to build the request document:

# COMMAND ----------

documents = {'documents' : [
    {'id': '1', 'language': 'en', 'text': claim_text}
]}

# COMMAND ----------

# MAGIC %md Now invoke the Text Analytics API and observe the result. 

# COMMAND ----------

headers   = {"Ocp-Apim-Subscription-Key": text_analytics_subscription_key}
response  = requests.post(sentiment_api_url, headers=headers, json=documents)
sentiments = response.json()
sentiments

# COMMAND ----------

# MAGIC %md To parse out the sentiment score from the response, run the following cell:

# COMMAND ----------

score = sentiments['documents'][0]['score']
score

# COMMAND ----------

# MAGIC %md You can provide a human-friendly interpretation on this score by running the following cell:

# COMMAND ----------

score_interpretation = "neutral"
if (score < 0.45): 
    score_interpretation = "negative"
elif (score >= 0.55):
    score_interpretation = "positive"
score_interpretation

# COMMAND ----------

# MAGIC %md ## Task 4 - Invoking the Azure ML Deployed Services

# COMMAND ----------

# MAGIC %md Run the following cell to define a method that will be used to invoke your classifier and summarizer methods deployed using Azure Machine Learning service to Azure Container Instances:

# COMMAND ----------

def invoke_service(ml_service_key, ml_service_scoring_endpoint, ml_service_input):
    headers   = {"Authorization": "Bearer " + ml_service_key}
    response  = requests.post(ml_service_scoring_endpoint, headers=headers, json=ml_service_input)
    result = response.json()
    return result

# COMMAND ----------

# MAGIC %md Configure the classifier invocation with the key and endpoint as appropriate to your deployed instance:

# COMMAND ----------

classifier_service_key = "" #leave this value empty if the service does not have authentication enabled
classifier_service_scoring_endpoint = "http://104.40.31.31:80/score"
classifier_service_input = claim_text

# COMMAND ----------

# MAGIC %md Invoke the classifier and observe the result:

# COMMAND ----------

classifier_result = invoke_service(classifier_service_key, classifier_service_scoring_endpoint, classifier_service_input)
classifier_result

# COMMAND ----------

# Interpret the classifier result
import json
classification = json.loads(classifier_result)
classification = 'Auto Insurance Claim' if classification == 1 else 'Home Insurance Claim' 
classification

# COMMAND ----------

# MAGIC %md Similarly, configure the key and scoring endpoint as appropriate to your summarizer service:

# COMMAND ----------

summarizer_service_key = "" #leave this value empty if the service does not have authentication enabled
summarizer_service_scoring_endpoint = "http://104.40.28.68:80/score"
summarizer_service_input = claim_text

# COMMAND ----------

# MAGIC %md Invoke the summarizer service and observe the result:

# COMMAND ----------

summarizer_result = invoke_service(summarizer_service_key, summarizer_service_scoring_endpoint, summarizer_service_input)
summarizer_result =  summarizer_result[0].replace("\\n", "").strip() if len(summarizer_result) > 0 else "N/A"
summarizer_result

# COMMAND ----------

# MAGIC %md ## Task 4 -  Summarizing the Results

# COMMAND ----------

# MAGIC %md In this final task, you pull together all of the pieces to display the results of your AI based processing.
# MAGIC 
# MAGIC Run the following cell and examine the result.

# COMMAND ----------

displayTemplate = """
<div><b>Claim Summary</b></div>
<div>Classification: {}</div>
<div>Caption: {}</div>
<div>Tags: {}</div>
<div>Text in Image: {}</div>
<div>Sentiment: {}</div>
<div><img src='{}' width='200px'></div>
<div>Summary: </div>
<div><pre>{} </pre></div>
<div>&nbsp;</div>
<div>Claim:</div>
<div>{}</div>

"""
displayTemplate = displayTemplate.format(classification, caption, ' '.join(topTags), ' '.join(words_list), 
                                         score_interpretation, image_url, summarizer_result, 
                                         claim_text)
displayHTML(displayTemplate)