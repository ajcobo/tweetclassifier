from load import *
import pandas as pa
import numpy as np
#dates
import datetime

def extract_features_with_timegap(location, event_date = False):
  dataframe = extract_dataframe_from_csv(location)
  
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

def extract_noise_features_with_timegap(location, base_first_date, event_date = False):
  dataframe = extract_dataframe_from_csv(location)
  
  if event_date is not False:
    dataframe = add_noise_time_gap(dataframe, event_date, base_first_date)
  

  return dataframe

def add_noise_time_gap(dataframe, event_date, base_first_date):
  #order by date
  dataframe.features = dataframe.features.sort(["created_at"])

  times = pa.to_datetime(dataframe.features["created_at"], unit="s")
  dataframe_first_date = pa.to_datetime(dataframe.features["created_at"][1], unit="s")

  # translate
  dataframe_gap = times - dataframe_first_date
  translated_times = dataframe.features['translated_created_at'] =  base_first_date + dataframe_gap

  # time gap
  time_gap = translated_times - event_date



  #Add time_gap for the features
  dataframe.features['time_gap'] = time_gap

  return dataframe


