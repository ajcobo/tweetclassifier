from load import *
#pandas
import pandas as pa
#dates
import datetime

def extract_features(location, event_date = False):
  dataframe = extract_dataframe_from_csv("/home/alf/Dropbox/Master/AC/Ground Truth/Consolidated/GroundTruthTotal.csv")
  
  if event_date is not False:
    dataframe = add_time_gap(dataframe, event_date)
  
  return dataframe

def add_time_gap(dataframe, event_date):
  times = pa.to_datetime(dataframe.features["date"] +" "+dataframe.features["time"])
  time_gap = times - event_date

  #order by date
  dataframe.features = dataframe.features.sort(['date', 'time'])

  #Add time_gap for the features
  dataframe.features['time_gap'] = time_gap

  return dataframe