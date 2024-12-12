# %% [markdown]
# # Air quality: an Exploratory Data Analysis Project
# This repository contains an exploratory data analysis (EDA) project focused on understanding air quality trends and their relationship with weather conditions. Using the Air Quality Dataset from the UCI Machine Learning Repository, the project applies data cleaning, visualization, and statistical techniques to uncover patterns and insights.

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# - 1 Loading Data
aiq_data = pd.read_csv('air+quality_ds/AirQualityUCI.csv', delimiter=';')
# Data source: https://archive.ics.uci.edu/dataset/360/air+quality

# Loading data info from the source website
data_info_raw = pd.read_html('https://archive.ics.uci.edu/dataset/360/air+quality')[0]


# %%
# - Backup
aiq_data_bk = aiq_data.copy()

# - First look
print(pd.concat([aiq_data.head(10), aiq_data.tail(10)]))

# %%
aiq_data.info()
aiq_data.describe().T

# %% [markdown]
# ## Data cleaning
# 
# ### Step 1: Clean the Data
# 
# - Convert decimal commas to dots.
# - Remove unnecessary columns.
# - Ensure all numeric columns are properly converted.

# %%
# - 2 Cleaning Data
# The last rows are completely empty, so we can remove them
aiq_data = aiq_data.dropna(how='all')

# %%
# - 2 Data Cleaning: Proper conversion of the data types and commas, numeric value etc.
# checking that every value has the correct data type
aiq_data.dtypes

# %%
# converting the date  to datetime
     # The date has the format of dd/mm/yyyy
aiq_data['Date'] = pd.to_datetime(aiq_data['Date'], format='%d/%m/%Y')
aiq_data['Time'] = pd.to_datetime(aiq_data['Time'], format='%H.%M.%S')

# %%
# converting non numeric columns to numeric 
# columns to be converted: ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']


# %%
# deleting unnecessary columns
# deleting column 15 and 16 from aiq_data dataframe if they exist
columns_to_drop = ['Unnamed: 15', 'Unnamed: 16']
aiq_data.drop(columns=[col for col in columns_to_drop if col in aiq_data.columns], inplace=True)


# %%
# a quick look to the df
aiq_data.head()

# %%
aiq_data.hist(bins=128, figsize=(20,10))
plt.show()

# %%
# To gather more information about the clipping values: we analyze the lower value for each column:
numeric_columns = aiq_data.select_dtypes(include=[np.number]).columns
aiq_data[numeric_columns].describe().T[['min']].sort_values(by='min', ascending=True).head(10)


# %%
# The clipping value is '-200' for every column. We will replace these values with NaN.
aiq_data.replace(-200, np.nan, inplace=True)

# count the number of NaN for each column
aiq_data.isnull().sum()
number_of_missing_values = aiq_data.isnull().sum().sum()
total_number_of_values = aiq_data.shape[0] * aiq_data.shape[1] 
print(f'The number of missing values in the dataset is: {number_of_missing_values} on overall {total_number_of_values} values')
print('with a percentage of missing data of: {:.2f}%'.format(number_of_missing_values / total_number_of_values * 100)) 


# %%
sb.heatmap(aiq_data.isnull(), cbar=False)



# %%
