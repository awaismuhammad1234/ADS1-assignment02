#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting environment to ignore future warnings
import warnings
warnings.simplefilter('ignore')


# In[2]:


def ingest_data(filename):
    """
    Read the World Bank dataset and return two dataframes:
    one with years as columns and one with countries as columns.
    """
    # Read dataset
    df = pd.read_csv(filename, skiprows=4).iloc[:, :-1]
    
    # Transform dataset with years as columns
    df_years = df

    # Transform dataset with countries as columns
    df_countries = df.set_index('Country Name').T
    df_countries.index.name = 'Indicator'
    df_countries = df_countries.reset_index()

    return df_years, df_countries


# In[3]:
