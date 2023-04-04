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


# Let's read data
filename = "API_19_DS2_en_csv_v2_4902199.csv"
df_years, df_countries = ingest_data(filename)
display(df_years.head())
display(df_countries.head())


# In[4]:


def explore_data(df, indicators, countries):
    """
    Explore the statistical properties of the selected indicators
    and countries, and return summary statistics.
    """
    # Filter dataset by selected countries
    selected_data = df[df['Country Name'].isin(countries)]
    
    summaries = {}
    for ind in indicators:
        # Filter dataset by the current indicator
        selected = selected_data[selected_data["Indicator Name"] == ind]
        
        # Drop unnecessary columns
        selected.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1, inplace=True)
        
        # Set the index to 'Country Name', transpose the dataframe, and calculate summary statistics
        summary = selected.set_index("Country Name").T.describe()
        
        # Add summary statistics for the current indicator to the summaries dictionary
        summaries[ind] = summary

    return summaries


# In[5]:


# Define indicators and countries of interest
indicators_of_interest = [
    'Population growth (annual %)',
    'CO2 emissions (metric tons per capita)',
    'Energy use (kg of oil equivalent per capita)',
    'Renewable energy consumption (% of total final energy consumption)'
]
countries_of_interest = [
    'United States', 'China', 'India', 'Germany', 'Brazil']

# Explore statistical properties of selected indicators and countries
summaries = explore_data(
    df_years, indicators_of_interest, countries_of_interest)

for ind, sumary in summaries.items():
    print(f"Statistical Characteristics of Selected Countries for Indicator {str(ind).upper()} are...")
    display(sumary)
