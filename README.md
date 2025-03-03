# CarCalcX
A linear regression model created using Sklearn, pandas and numpy to predict the car prize using parameters such as Brand, Fuel type, age etc.


# Libraries used and Workflow  

pandas: For data analysis , used https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset from kaggle  
Scikit-Learn: To train a linear regression model  
Numpy: To use log transformation on large price values in the dataset   
Pickle (with AI): To save the model  
Streamlit: Used Streamlit to deploy the model but due to some technical issues the model has not been completely deployed.  


# Additional changes  
Used feature engineering to reduce RMSE,MSE,R(square) on the model  
📊 Model Evaluation AFTER Feature Engineering:  

MAE  : 0.32  

MSE  : 0.19  

RMSE : 0.44  

R² Score : 0.7306    

# STEPS TO USE THE MODEL  
STEP 1: Download the dataset
STEP 2: Download all the reuqired libraries
STEP 3: Run the two files 1. data_gathering.ipynb and 2. final_dataset.ipynb in jupyter notebook  
STEP 4: using app.py type the command in terminal: streamlit run app.py  
STEP 5: You will see the web interface follow further commnands to use the model.  
