import numpy as np
from flask import Flask, abort, jsonify, request
from topic_modeling import Topic
import tensorflow as tf
#from sklearn.externals import joblib
#from utils import build_path
#import nltk
import json
#import time
import pandas as pd
#import csv
app = Flask(__name__)
#Creating Topic modeling object
topic_obj = Topic()
#topic_obj.trainPath()
topic_obj.reloadModelAndData()
print("Topic modeling model restored")
@app.route('/topic_modeling/welcome', methods = ['GET'])
def index():
        return "Welcome to ABCNN!"
@app.route('/topic_modeling/retrain', methods = ['GET'])
def retrain_models():
        tf.reset_default_graph()
        topic_obj.trainPath()
        topic_obj.reloadModelAndData()
        return "Retrainig Done!!"
@app.route('/topic_modeling/query', methods = ['POST'])
def Topic_modeling():
        #Extract data from request
        data=request.get_json(force=True)
        prob = data["question"]
        sys_id = data["sys_id"]
        print(prob)
        print(sys_id)
        #Topic Modeling
        problem = str(prob).lower().strip()
        topic_obj.response(problem)
        sols = topic_obj.solutions
        problem_cat = topic_obj.topic
        print(problem_cat)
        return jsonify(results = {problem:problem_cat[0][0]})
if __name__ == "__main__":
        app.run(host='0.0.0.0', port=9000)
                                        
