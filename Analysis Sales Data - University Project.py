#!/usr/bin/env python
# coding: utf-8

# # Sales Analysis

# I use Python Pandas & Python Matplotlib to analyze and answer business questions about 12 months worth of sales data. The data contains hundreds of thousands of electronics store purchases broken down by month, product type, cost, purchase address, etc. 
# I start by cleaning our data. Tasks during this section include:
# - Drop NaN values from DataFrame
# - Removing rows based on a condition
# - Change the type of columns (to_numeric, to_datetime, astype)
# 
# Once I have cleaned up our data a bit, I move the data exploration section. In this section I explore 5 high level business questions related to our data:
# - What was the best month for sales? How much was earned that month?
# - What city sold the most product?
# - What time should we display advertisemens to maximize the likelihood of customer’s buying product?
# - What products are most often sold together?
# - What product sold the most? Why do you think it sold the most?
# 
# To answer these questions I walk through many different pandas & matplotlib methods. They include:
# - Concatenating multiple csvs together to create a new DataFrame (pd.concat)
# - Adding columns
# - Parsing cells as strings to make new columns (.str)
# - Using the .apply() method
# - Using groupby to perform aggregate analysis
# - Plotting bar charts and lines graphs to visualize our results
# - Labeling our graphs
# 

# #### Import necessary libraries

# In[29]:


import pandas as pd
import os
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# #### Merge data from each month into one CSV

# In[3]:


files = [file for file in os.listdir('/Users/apple/Downloads/AnalysisProject/SalesAnalysis/Sales_Data')]

all_months_data = pd.DataFrame()
for file in files:
    df = pd.read_csv("/Users/apple/Downloads/AnalysisProject/SalesAnalysis/Sales_Data/"+file)
    all_months_data = pd.concat([all_months_data, df])
    
all_months_data.to_csv('all_data.csv', index=False)


# #### Read in updated dataframe

# In[4]:


all_data = pd.read_csv("all_data.csv")
all_data.head()


# ### Clean up the data!
# The first step in this is figuring out what we need to clean. I have found in practice, that you find things you need to clean as you perform operations and get errors. Based on the error, you decide how you should go about cleaning the data

# ##### Drop rows of NAN

# In[5]:


# Find NAN

nan_df = all_data[all_data.isna().any(axis=1)]
display(nan_df.head())

all_data = all_data.dropna(how='all')
all_data.head()


# ##### Get rid of text in order date column

# In[6]:


all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']


# #### Make columns correct type

# In[7]:


all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])


# ### Augment data with additional columns

# #### Add month column

# In[8]:


all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')
all_data.head()


# #### Add a sales column
# 
# 
# 

# In[9]:


all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each'] 
all_data.head()


# #### Add city column

# In[10]:


def get_city(address):
    return address.split(",")[1].strip(" ")

def get_state(address):
    return address.split(",")[2].split(" ")[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)}  ({get_state(x)})")
all_data.head()



# ## Data Exploration!

# #### Question 1 : What was the best month for sales? How much was earned that month?

# In[11]:


all_data['Sales'] = all_data['Quantity Ordered'].astype('int') * all_data['Price Each'].astype('float')


# In[12]:


all_data.groupby(['Month']).sum(numeric_only=True)


# In[20]:


import matplotlib.pyplot as plt

months = all_data['Month'].unique()

plt.bar(months,all_data.groupby(['Month']).sum(numeric_only=True)['Sales'])
plt.xticks(months)
plt.ylabel('Sales in USD ($)')
plt.xlabel('Month number')
plt.show()


# #### Question 2: What city had the highest number of sales?

# In[17]:


results = all_data.groupby('City').sum(numeric_only=True)
results


# In[16]:


import matplotlib.pyplot as plt

keys = [city for city, df in all_data.groupby('City')]

plt.bar(keys,all_data.groupby(['City']).sum(numeric_only=True)['Sales'])
plt.ylabel('Sales in USD ($)')
plt.xlabel('City Name')
plt.xticks(keys, rotation='vertical', size=8)
plt.show()


# #### Question 3: What time should we display advertisements to maximize likelihood of customer's buying product?

# In[21]:


# Add hour column
all_data['Hour'] = pd.to_datetime(all_data['Order Date']).dt.hour
all_data['Minute'] = pd.to_datetime(all_data['Order Date']).dt.minute
all_data['Count'] = 1
all_data.head()


# In[23]:


keys = [pair for pair, df in all_data.groupby('Hour')]

plt.plot(keys, all_data.groupby('Hour').count()['Count'])
plt.xticks(keys)
plt.xlabel('Hour')
plt.ylabel('Number Of Orders')
plt.grid()
plt.show()

# My recommendation is slightly before 11am or 7pm


# #### Question 4: What products are most often sold together?

# In[69]:


# https://stackoverflow.com/questions/43348194/pandas-select-rows-if-id-appear-several-time

df = all_data[all_data['Order ID'].duplicated(keep=False)]

# Referenced: https://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
df = df[['Order ID', 'Grouped']].drop_duplicates()

df.head()


# In[70]:


# Referenced: https://stackoverflow.com/questions/52195887/counting-unique-pairs-of-numbers-into-a-python-dictionary

from itertools import combinations
from collections import Counter

count = Counter()

for row in df2['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

for key,value in count.most_common(10):
    print(key, value)


# #### What product sold the most? Why do you think it sold the most?

# In[71]:


all_data.head()


# In[25]:


product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum(numeric_only=True)['Quantity Ordered']

products = [product for product, df in product_group]
plt.bar(products, quantity_ordered)
plt.xticks(keys, rotation='vertical', size=8)
plt.xlabel('Product')
plt.ylabel('Quantity Ordered')
plt.show()


# In[32]:


# Referenced: https://stackoverflow.com/questions/14762181/adding-a-y-axis-label-to-secondary-y-axis-in-matplotlib
import matplotlib.pyplot as plt

prices = all_data.groupby('Product').mean(numeric_only=True)['Price Each']
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices, color='b')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price ($)', color='b')
ax1.set_xticks(range(len(products)))
ax1.set_xticklabels(products, rotation='vertical', size=8)

fig.show()


# In[ ]:




