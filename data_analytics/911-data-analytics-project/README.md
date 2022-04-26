# 911 Calls Data Analysis


In this project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). 

The code is in the 911-Data Analysis.ipynb file.

## Objectives
### Analyse the data and get below insights
- Top titles, zipcodes, townships for 911 calls.
- Most common reason for calls.
- Compare the count of the reasons using countplot.
- Compare the count of incident over the week and month using different seaborn plots.


## Tech Stack
- **Python modlues**: pandas, numpy, seaborn and matplotlib.
- **Basic analysis function/methods**: values_counts() and nunique()
- **Transformations**: apply(), groupby(), reset_index(), to_datetime(), map() and unstack()
- **Plots**: countplot(), lmplot(), heatmap() and clustermap()


## Metadata
* lat : String variable, Latitude
* lng: String variable, Longitude
* desc: String variable, Description of the Emergency Call
* zip: String variable, Zipcode
* title: String variable, Title
* timeStamp: String variable, YYYY-MM-DD HH:MM:SS
* twp: String variable, Township
* addr: String variable, Address
* e: String variable, Dummy variable (always 1)