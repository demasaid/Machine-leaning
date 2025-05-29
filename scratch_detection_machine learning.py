#!/usr/bin/env python
# coding: utf-8

# # Start work

# In[ ]:


In this assignment you are receiving wafer maps in a certain operation and the goal is to predict whether a given die belongs to scratch or not.

The data includes information about individual dies from a number of wafers.

The table data includes the following columns:

WaferName : The name of the wafer from which the die came.
DieX: The horizontal position of the die on the wafer.
DieY: The vertical position of the die on the wafer.
IsGoodDie: A binary column indicating whether the die is good or not.
IsScratchDie: A binary column indicating whether the die belongs to a scratch or not.
Your goal is to use the training data to build a model that can predict, given a certain wafer map, the dies on the map that are parts of a scratch (whether they are bad, 'Scratch' or good, 'Ink').

The purpose of the assignment is mainly to get to reasonable solution that can help the business. Please note that real industry solutions usually achieve lower scores than you may be used from academic problems so even a low metric score on the test set may be considered a success

Business goals:

Automation. This process is currently a manual and expensive procedure that takes a lot of time and is prone to errors by the tagger. The goal is to perform this procedure in a faster time and save the costs of the test
Quality. increasing the quality of the dies while balancing quality and yield (on the one hand, not to miss scratches, on the other hand not to do too much "Ink")
Prediction Level. As explained above, the main goal is to detect individual dies, but sometimes it will help to also get a classification at the wafer level, (binary classification, is there a scratch on this wafer or not?) because there are manufacturers who return scratched wafers to the factory.
Note. In wafers with a low yield (that is, a lot of faulty dies), we will not perform scratch detection because the customer is afraid to find randomly generated scratches there and perform unnecessary ink. In such cases, the customer will make sure to check all the dies strictly in any case, but regardless of the detection of scratches. Therefore, in these cases we will not consider a sequence of bad die to be scratch.

You are free to use any machine learning technique you find appropiate for solving this problem. Make sure choosing the relevamt metrics to test your solutions's performance.

In addition to the training data, you are given a test set, which includes the x and y coordinates and the good/not status of each die, but does not include the scratch/not scratch labels.

You are asked to use your model to predict the scratch/not scratch status of the dies in the test set, and to save the predictions in a CSV file. You should submit your notebook including the experiments you did along the way to improve the model/various methods you tried and including your final model.

Pay attention to the following points:

Exploratoration and analyze the data
Consideration of business goals
Selection of relevant machine learning models
Appropriate choice of metrics


# In[48]:


import pandas as pd
import zipfile
from datetime import datetime


# ### Load Data

# In[49]:


#load zip file
zf = zipfile.ZipFile('data.zip') 


# In[50]:


#load train data
df_wafers = pd.read_csv(zf.open('wafers_train.csv'))
df_wafers.head()


# In[51]:


#load test data
df_wafers_test = pd.read_csv(zf.open('wafers_test.csv'))
df_wafers_test.head()


# You can draw the wafers map to see how the wafers look like in the data. 
# 
# Using the following helper function you can draw the wafer maps with or without labels:

# In[52]:


def plot_wafer_maps(wafer_df_list, figsize, labels = True):
    """
    plot wafer maps for list of df of wafers

    :param wafer_df_list: list, The list of df's of the wafers
    :param figsize: int, the size of the figsize height 
    :param labels: bool, Whether to show the layer of labels (based on column 'IsScratchDie')
    
    :return: None
    """
    def plot_wafer_map(wafer_df, ax, map_type):
        wafer_size = len(wafer_df)
        s = 2**17/(wafer_size)
        if map_type == 'Label':
            mes = 'Scratch Wafer' if (wafer_df['IsScratchDie'] == True).sum()>0 else 'Non-Scratch Wafer'
        else:
            mes = 'Yield: ' + str(round((wafer_df['IsGoodDie']).sum()/(wafer_df['IsGoodDie']).count(), 2)) 
        
        ax.set_title(f'{map_type} | Wafer Name: {wafer_df["WaferName"].iloc[0]}, \nSum: {len(wafer_df)} dies. {mes}', fontsize=20)
        ax.scatter(wafer_df['DieX'], wafer_df['DieY'], color = 'green', marker='s', s = s)

        bad_bins = wafer_df.loc[wafer_df['IsGoodDie'] == False]
        ax.scatter(bad_bins['DieX'], bad_bins['DieY'], color = 'red', marker='s', s = s)
        
        if map_type == 'Label':
            scratch_bins = wafer_df.loc[(wafer_df['IsScratchDie'] == True) & (wafer_df['IsGoodDie'] == False)]
            ax.scatter(scratch_bins['DieX'], scratch_bins['DieY'], color = 'blue', marker='s', s = s)

            ink_bins = wafer_df.loc[(wafer_df['IsScratchDie'] == True) & (wafer_df['IsGoodDie'] == True)]
            ax.scatter(ink_bins['DieX'], ink_bins['DieY'], color = 'yellow', marker='s', s = s)

            ax.legend(['Good Die', 'Bad Die', 'Scratch Die', 'Ink Die'], fontsize=8)
        else:
            ax.legend(['Good Die', 'Bad Die'], fontsize=8)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False) 
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if labels:
        fig, ax = plt.subplots(2, len(wafer_df_list), figsize=(figsize*len(wafer_df_list), figsize*2))
        for idx1, wafer_df in enumerate(wafer_df_list):
            for idx2, map_type in enumerate(['Input', 'Label']):
                plot_wafer_map(wafer_df, ax[idx2][idx1], map_type)
    else:
        fig, ax = plt.subplots(1, len(wafer_df_list), figsize=(figsize*len(wafer_df_list), figsize))
        for idx, wafer_df in enumerate(wafer_df_list):
            plot_wafer_map(wafer_df, ax[idx], 'Input')

    plt.show()


# Select the amount of samples you want to display:

# In[53]:


n_samples = 4
list_sample_train = [df_wafers.groupby('WaferName').get_group(group) for group in df_wafers['WaferName'].value_counts().sample(n_samples, random_state=20).index]
plot_wafer_maps(list_sample_train, figsize = 8, labels = True)


# In[54]:


list_sample_test = [df_wafers_test.groupby('WaferName').get_group(group) for group in df_wafers_test['WaferName'].value_counts().sample(n_samples, random_state=20).index]
plot_wafer_maps(list_sample_test, figsize = 8, labels = False)


# # Build your solution

# In[55]:


print(df_wafers.info())
print(df_wafers.describe())
print(df_wafers.isnull().sum())


# In[56]:


#chick if there a missing value 
print("Missing values per column:")
print(X_train.isnull().sum())


# In[57]:


# chick if there are duplicate rows , I whene i remove it i got a negative effect so i kept it   
print("Number of duplicated rows:", X_train.duplicated().sum())


# In[58]:


#i calculate the precent of dies with scratch so now i have view in the data that i have imbalanced data  
scratch_percentage = df_wafers['IsScratchDie'].mean() * 100
print(f"Percentage of dies with scratch: {scratch_percentage:.2f}%")


# In[62]:


scratch_dies = df_wafers[df_wafers['IsScratchDie'] == True]

plt.figure(figsize=(4,4))
sns.scatterplot(data=scratch_dies, x='DieX', y='DieY', hue='IsGoodDie')
plt.title('Scratch Dies Positions')
plt.xlabel('DieX')
plt.ylabel('DieY')
plt.show()


# In[60]:


#use a validation set to evaluate the model’s performance on unseen data 

from sklearn.model_selection import train_test_split

# Convert boolean values to integers 0 or 1 
df_wafers['IsGoodDie'] = df_wafers['IsGoodDie'].astype(int)
df_wafers['IsScratchDie'] = df_wafers['IsScratchDie'].astype(int)


features = ['DieX', 'DieY', 'IsGoodDie']
X = df_wafers[features]
y = df_wafers['IsScratchDie']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")


# In[63]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt




# A - Train Balanced Random Forest

print("Training Balanced Random Forest")
brf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf_model.fit(X_train_split, y_train_split)
brf_pred = brf_model.predict(X_val_split)

print("Balanced Random Forest Evaluation:")
print(classification_report(y_val_split, brf_pred))


# B Train XGBoost Classifier


print("Training XGBoost")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_split, y_train_split)
xgb_pred = xgb_model.predict(X_val_split)

print("XGBoost Evaluation")
print(classification_report(y_val_split, xgb_pred))


# C- Train Random Forest + SMOTE

print("Applying SMOTE")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_split, y_train_split)

print("\nTraining Random Forest (after SMOTE)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
rf_pred = rf_model.predict(X_val_split)

print("Random Forest with SMOTE Evaluation:")
print(classification_report(y_val_split, rf_pred))


# D-  Train Autoencoder for Anomaly Detection


print("Training Autoencoder")


X_train_autoencoder = X_train_cleaned[y_train_cleaned == 0]

# Further split Good Dies into train/validation
X_train_auto, X_val_auto = train_test_split(X_train_autoencoder, test_size=0.2, random_state=42)

input_dim = X_train_auto.shape[1]

# Define Autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train Autoencoder
history = autoencoder.fit(
    X_train_auto, X_train_auto,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val_auto, X_val_auto),
    verbose=1
)

# Predict reconstruction for validation set
X_val_reconstructed = autoencoder.predict(X_val_split)
# Compute reconstruction error (MSE)
reconstruction_errors = np.mean(np.square(X_val_split - X_val_reconstructed), axis=1)
# Set threshold (e.g., 95th percentile)
threshold = np.percentile(reconstruction_errors, 95)

print("Autoencoder Threshold for Anomaly Detection:", threshold)

# Predict anomalies based on reconstruction error
autoencoder_predictions = (reconstruction_errors > threshold).astype(int)

print("Autoencoder Anomaly Detection Evaluation:")
print(classification_report(y_val_split, autoencoder_predictions))



# In[37]:


from sklearn.model_selection import GridSearchCV


xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}


grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='recall',  # because we care more about Recall for scratches
    cv=3,
    verbose=2,
    n_jobs=-1
)


grid_search.fit(X_train_split, y_train_split)
print("\nBest Parameters Found:")
print(grid_search.best_params_)

# Train best model
best_xgb = grid_search.best_estimator_

# Predict on validation set
y_pred = best_xgb.predict(X_val_split)

# Evaluate
print("\nTuned XGBoost Evaluation:")
print(classification_report(y_val_split, y_pred))


# # Additional thoughts

# I used 4 different models in this project:
# 
# A-Train Balanced Random Forest  might  be a good choice for handling imbalanced data 
# 
# B-Random Forest with SMOTE:Because the data had very few samples labeled as scratch (label = 1) we used SMOTE to intelligently create synthetic examples and help the model learn better from imbalanced data.
# 
# C-XGBoost Classifier:XGBoost was chosen because it is very efficient and naturally handles imbalanced data well,It is also faster to train 
# 
# D-Autoencoder (using TensorFlow):I experimented with an Autoencoder model because it is designed to detect anomalies  in the data.
# 
# ((TensorFlow was selected as the deep learning framework for better integration and performance compared to' PyTorch' for this simple setup.))
# 
# Results and Model Selection:
# 
# After evaluating all models, the best results came from the XGBoost model.

# So I choose  XGBoost model and , my thought are to  optimization To improve the performance further I performed hyperparameter optimization on XGBoost.
# 

# Here you can detail about anything you want to mention as additional considerations for this solution, anything from ideas, thoughts, considerations about deployment or anything you may have raised when working on this task in a team.

# In[64]:


#TODO the thoughts

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

# Define the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Set up Grid Search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='recall',  # because we care more about Recall for scratches
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Perform Grid Search on training data
grid_search.fit(X_train_split, y_train_split)

# Best parameters
print("\nBest Parameters Found:")
print(grid_search.best_params_)

# Train best model
best_xgb = grid_search.best_estimator_

# Predict on validation set
y_pred = best_xgb.predict(X_val_split)

# Evaluate
print("\nTuned XGBoost Evaluation:")
print(classification_report(y_val_split, y_pred))


# After hyperparameter tuning using GridSearchCV XGBoost achieved a recall of 34% and precision of 42% for scratch detection, showing'''''significant'''' improvement over the initial models

# # Submission

# In[65]:


#TODO

# Train the final model (Tuned XGBoost)
final_model = XGBClassifier(
    colsample_bytree=1,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=200,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

final_model.fit(X_train_cleaned, y_train_cleaned)

# Drop 'WaferName' and 'IsScratchDie' if they exist from test set
columns_to_drop = ['WaferName']
if 'IsScratchDie' in df_wafers_test.columns:
    columns_to_drop.append('IsScratchDie')

df_wafers_test_cleaned = df_wafers_test.drop(columns=columns_to_drop)

#model = (XGBoost)
IsScratchDie_pred = final_model.predict(df_wafers_test_cleaned)
df_wafers_test['IsScratchDie'] = IsScratchDie_pred

#TODO Fill in your name and email
name = 'Dema Saed'
email = 'demasaed.21@gmail.com'

#Dont change the following code
date_str = datetime.now().strftime('%Y%m%d')
filename = f"{date_str}_{name}_{email}_df_wafers_test_with_preds.csv"
df_wafers_test.to_csv(filename, index=False)
print("Saved file:", filename)


# In[47]:


# Import datetime
from datetime import datetime
from imblearn.ensemble import BalancedRandomForestClassifier  # make sure you have this import too

# Train the final model
final_model = BalancedRandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    max_features='sqrt', 
    random_state=42
)

final_model.fit(X_train_cleaned, y_train_cleaned)

# Drop the 'WaferName' column because model expects numeric features only
# Drop both 'WaferName' and 'IsScratchDie' (if it exists)
columns_to_drop = ['WaferName']
if 'IsScratchDie' in df_wafers_test.columns:
    columns_to_drop.append('IsScratchDie')

df_wafers_test_cleaned = df_wafers_test.drop(columns=columns_to_drop)


#model = (Balanced Random Forest)
IsScratchDie_pred = final_model.predict(df_wafers_test_cleaned)
df_wafers_test['IsScratchDie'] = IsScratchDie_pred

#TODO Fill in your name and email
name = 'Dema Saed'
email = 'demasaed.21@gmail.com'

#Dont change the following code
date_str = datetime.now().strftime('%Y%m%d')
filename = f"{date_str}_{name}_{email}_df_wafers_test_with_preds.csv"
df_wafers_test.to_csv(filename, index=False)
print("Saved file:", filename)

