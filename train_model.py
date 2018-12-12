
import pickle
import scipy
from lightfm import LightFM
from lightfm.data import Dataset

import boto3
import pandas as pd
import numpy as np
from random import randint

def create_recommender():
	# obtain interaction table from dynamodb, which is json data
	dynamodb = boto3.resource('dynamodb')
	table = dynamodb.Table('eye_video_vote')
	response = table.scan()
	raw_data = response['Items']
	
		
	#transform json structure data to user-item-rating interaction format
	final_df = pd.DataFrame(columns=['userId','videoId','rating'])

	for i in raw_data:
#    data = raw_data[i]
		if any('upVote' in s for s in list(i.keys())):
			df1 = {k:i[k] for k in ('upVote','videoId')}
			df1['videoId'] = {df1['videoId']}
			df1 = pd.DataFrame.from_dict(df1, orient='index').T
			df1['rating'] = randint(4, 5)
			df1.fillna(value = pd.np.nan, inplace=True)
			df1 = df1.fillna(method='ffill')
			df1.rename(columns={'upVote':'userId'},inplace=True)
			final_df = final_df.append(df1)
		if any('downVote' in s for s in list(i.keys())):
			df2 = {k:i[k] for k in ('downVote','videoId')}
			df2['videoId'] = {df2['videoId']}
			df2 = pd.DataFrame.from_dict(df2, orient='index').T
			df2['rating'] = randint(1, 2)
			df2.fillna(value = pd.np.nan, inplace=True)
			df2 = df2.fillna(method='ffill')
			df2.rename(columns={'downVote':'userId'},inplace=True)
			final_df = final_df.append(df2)
	
	
	#rename the columns 
	final_df.rename(columns={'userId':'UserID', 'videoId':'MovieID', 'rating':'rating'}, inplace=True)
		
	#generate the appropriate lightfm dataset
	dataset = Dataset()
	dataset.fit(users = (row['UserID'] for index,row in final_df.iterrows()),
				items = (row['MovieID'] for index,row in final_df.iterrows()))
	
	(interactions, weights) = dataset.build_interactions((row['UserID'],row['MovieID'],row['rating']) for index,row in final_df.iterrows())
	
	#model collabrative filtering
	model_cf = LightFM(no_components=20, loss='warp')
	model_cf.fit(interactions, user_features=None, item_features=None, sample_weight=None, epochs=20, num_threads=4)
	with open('model_cf.pickle', 'wb') as fle:
		pickle.dump(model_cf, fle, protocol=pickle.HIGHEST_PROTOCOL)
	
	return 

if __name__ == '__main__':
	create_recommender()

