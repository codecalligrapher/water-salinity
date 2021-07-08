import datetime
import csv
import pandas as pd
from tqdm import tqdm
import ee
import os


# REACHED 45880
# REACHED 67646
ee.Initialize()

RESULTS_PATH = '../../results/equinox'

df = pd.read_csv(os.path.join(RESULTS_PATH, 'df_latlon.csv'))[113526:]

IMAGE_COLLECTION = 'HYCOM/sea_temp_salinity'

file = open("../../data/HYCOM3.csv", "a", newline='')
writer = csv.writer(file)

# write headers
writer.writerow(['start_date', 'lat', 'lon', 'water_temp_0', 'salinity_0' ,'water_temp_2','salinity_2','water_temp_4','salinity_4','water_temp_4' ,'salinity_4','water_temp_6', 'salinity_6', 'water_temp_8', 'salinity_8'])

data = []

for index, row in tqdm(df.iterrows()):
    start_date = str(row['DATE_UTC'])
    end_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    lat, lon = row['LAT'], row['LONG']
    point = ee.Geometry.Point([lon, lat])
    
    collection = ee.ImageCollection(IMAGE_COLLECTION).filter(ee.Filter.geometry(point)).filter(ee.Filter.date(start_date, end_date))
    size = collection.size().getInfo()    
    collection = collection.toList(size)
    try:
        for idx in range(size):

            
            image = ee.Image(collection.get(idx)).multiply(0.001).add(20)
            band_info = image.reduceRegion(ee.Reducer.mean(), geometry=point).getInfo()
            
            water_temp_0 = band_info['water_temp_0']
            salinity_0 = band_info['salinity_0']
            water_temp_2 =  band_info['water_temp_2']
            salinity_2 =  band_info['salinity_2']
            water_temp_4 =  band_info['water_temp_4']
            salinity_4 =  band_info['salinity_4']
            water_temp_6 =  band_info['water_temp_6']
            salinity_6 = band_info['salinity_6']
            water_temp_8 =  band_info['water_temp_8']
            salinity_8 =  band_info['salinity_8']               
            writer.writerow([start_date, lat, lon, water_temp_0, salinity_0 ,water_temp_2,salinity_2,water_temp_4,salinity_4,water_temp_4 ,salinity_4,water_temp_6, salinity_6, water_temp_8, salinity_8])
            print(str([start_date, lat, lon, water_temp_0, salinity_0]))
            
    except Exception as e:
        print(f'Error: {e}')
        continue