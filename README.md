# Recommendation Engine API with lightFM

This is a repository demonstrating the bulid of a recommendation system using data directly from dynamodb.

The process consists 3 major steps
1. Load and transform json format data from dynamodb to user-item-rating format dataframe
2. Use transformed data to train the recommendation engine with lightFM and save the model as a pickle file
3. Create a flask API to call the finished model so it can be used on web service 
  - If new user, items with high average ratings for be recommended
  - If old user, items with be recommended using collabrotive filtering 
