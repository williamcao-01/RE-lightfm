import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy
from lightfm import LightFM
from lightfm.data import Dataset

app = Flask(__name__)

model_cf = pickle.load(open('model_cf.pickle', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
	num_rec = 15
	
	#total number of items
	rating_df = pd.read_csv('data/ratings.csv')
	num_items = len(set(list(rating_df['item'])))

	data = request.get_json(force = True)
	#number of ratings the user given
	user_id = data['user_id']
	
	num_interaction = rating_df[rating_df['user'] == user_id].shape[0]
	response = {}
		
	if num_interaction > 10:
		scores = pd.Series(model_cf.predict(user_ids = user_id, item_ids = np.arange(num_items)))
		scores = list(pd.Series(scores.sort_values(ascending=False).index))
		rec = list(map(int,scores[0:num_rec]))
		response['recommed'] = rec
		
		print(type(rec[0]))
		print(type(rec[1]))
		return jsonify(response)

	
	else :
		rating_df_grouped = rating_df.groupby('item', as_index=False, sort=False)['rating'].mean()
		scores = rating_df_grouped.sort_values(['rating'], ascending=0)[0:num_rec]
		scores = scores['item'].tolist()
		rec = scores[0:num_rec]
		response['recommed'] = rec
		print(type(rec[0]))
		print(type(rec[1]))
		return jsonify(response)
		
if __name__ == '__main__':
    app.run(port=9000, debug=False )	