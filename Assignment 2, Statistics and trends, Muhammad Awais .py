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


# In[6]:


def visualize_data(df, indicator, countries, title):
    """
    Visualize the data using line plots for the selected indicators and countries.
    """
    # Filter dataset by selected countries
    selected_data = df[df['Country Name'].isin(countries)]

    # Filter dataset by selected indicator and set year as index
    selected_data = selected_data[selected_data['Indicator Name'] == indicator]
  
    # Drop unnecessary columns
    selected_data = selected_data.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1).set_index("Country Name").T
    
    
    # Create line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=selected_data, dashes=False)
    plt.title(title)
    plt.ylabel(indicator)
    plt.xlabel('Year')
    plt.xticks(rotation=90)
    plt.legend(countries)
    plt.show()


# In[7]:


for indicator in indicators_of_interest:
        visualize_data(df_years, indicator, countries_of_interest,
                       f"{indicator} for selected countries")


# In[8]:


def visualize_relationship(df, ind_x, ind_y, countries):
    """
    Visualize the relationship between two indicators for the selected countries
    using scatter plots with regression lines.
    """
    
    # Create a figure and axes
    plt.figure(figsize=(12, 5))
    for i, country in enumerate(countries):
        x = df[(df["Country Name"] == country) & (df_years["Indicator Name"] == ind_x)].iloc[:, 4:].T
        y = df[(df["Country Name"] == country) & (df_years["Indicator Name"] == ind_y)].iloc[:, 4:].T
        
        x = np.array(x)
        y = np.array(y)
        
        sns.regplot(x, y, label=country)
    
    plt.xlabel(f"{ind_x} (Scaled)")
    plt.ylabel(f"{ind_y} (Scaled)")
    plt.title(f"Correlation between {ind_x} and {ind_y}")
    plt.legend()
    plt.show()


# In[9]:


# Visualize the relationship between population growth and CO2 emissions
visualize_relationship(df_years, "Population growth (annual %)", "CO2 emissions (metric tons per capita)", countries_of_interest)


# In[10]:


# Visualize the relationship between population growth and CO2 emissions
visualize_relationship(df_years, "Population growth (annual %)", "Renewable energy consumption (% of total final energy consumption)", countries_of_interest)


# In[11]:


# Visualize the relationship between population growth and CO2 emissions
visualize_relationship(df_years, "Population growth (annual %)", "Energy use (kg of oil equivalent per capita)", countries_of_interest)


# In[12]:


df = df_years[df_years['Country Name'].isin(countries_of_interest) & df_years['Indicator Name'].isin(indicators_of_interest)]

# Pivot the data to make it suitable for the heatmap
df_pivot = df.pivot(index='Country Name', columns='Indicator Name', values='2018')
corr = df_pivot.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create a heatmap of correlations between indicators
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlations between Climate Indicators')
plt.show()


# Next I'm going to build some plots using plotly library and the limitation of this library is that the graphs that are made with this library are not constant means when the kernel will be stopped all of the floors will disappear.

# In[13]:


import plotly.graph_objs as go

# Create a bubble chart of population, CO2 emissions, and GDP
fig = go.Figure(data=go.Scatter(x=df['2018'], y=df['2019'], mode='markers',
                                marker=dict(size=df['2017'].fillna(0)/100000000, sizemode='area', 
                                            sizeref=0.1, color=df['2019'], colorscale='Viridis', showscale=True),
                                text=df['Country Name']))

fig.update_layout(title='Population, CO2 Emissions, and GDP by Country',
                  xaxis_title='CO2 Emissions per Capita (metric tons)',
                  yaxis_title='GDP per Capita (constant 2010 US$)')

fig.show()


# In[14]:


import plotly.express as px

# Select the required indicators and countries
indicator = 'CO2 emissions (metric tons per capita)'
df = df_years[df_years['Indicator Name'] == indicator]

# Create a choropleth map of CO2 emissions
fig = px.choropleth(df, locations='Country Code', color='2018', 
                    title='CO2 Emissions per Capita by Country in 2018',
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()


# In[ ]:
