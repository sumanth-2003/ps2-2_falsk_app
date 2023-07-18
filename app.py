from flask import Flask, render_template, request, redirect, jsonify
# request: An object that represents the incoming HTTP request.
# redirect: A function for redirecting to a different URL.
from datetime import datetime#A module that provides classes for working with dates and times.
import io, urllib, base64
# io: A module that provides the fundamental file and stream classes.
# urllib: A module for handling URLs.
# base64: A module for encoding and decoding binary data using Base64 representation.
import matplotlib.pyplot as plt# A module for creating and manipulating plots and figures.
import matplotlib
import matplotlib.gridspec as gridspec # A module for specifying the geometry of the grid that a subplot will be placed on.
import pandas as pd #A library for data manipulation and analysis. It provides data structures like DataFrame for working with structured data.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor #A class for creating a random forest regressor model.
import numpy as np #A library for numerical computing in Python.
import odc.algo # A module from the Open Data Cube library that provides various algorithms for working with remote sensing data.
import plotly.io as pio
import plotly.graph_objs as go #A module that contains classes for creating and manipulating Plotly graphs and charts.

from geopy.geocoders import Nominatim #converting between addresses and geographic coordinates.

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
matplotlib.use('Agg')#With the 'Agg' backend, Matplotlib can render the plots and save them directly to a file, such as a PNG image
import datacube
# this code first creates a datacube object. This object provides access to the Datacube API. Then, the code uses the load() method to load the remote sensing data for the study area. The load() method takes a number of arguments, including the product name, the x and y coordinates of the study area, the time range, and the measurements that you want to load.
# Once the data is loaded, the code can then be used to perform analysis on the data. For example, you could use the ndvi() method to calculate the NDVI index for the study area.

app = Flask(__name__)#used to create a instance of flask app which is used to create apis
@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/type/<analysis_type>', methods=['POST'])
def analysis(analysis_type):
    if request.method=="POST":
        data = request.get_json()
        
        coordinates = data['coordinates']
        time_range = (data['fromdate'], data['todate'])
        study_area_lat = (coordinates[0][0], coordinates[1][0])
        study_area_lon = (coordinates[1][1], coordinates[2][1])

        try:
            dc = datacube.Datacube(app='water_change_analysis')

            ds = dc.load(product='s2a_sen2cor_granule',
                x=study_area_lon,
                y=study_area_lat,
                time=time_range,
                measurements=['red', 'green', 'blue', 'nir'],
                output_crs='EPSG:4326',
                resolution=(-0.00027, 0.00027)
            )
           
            ds = odc.algo.to_f32(ds)
            if analysis_type=="ndvi":
                res = (ds.nir - ds.red) / (ds.nir + ds.red)
            elif analysis_type=="ndwi":
                res = (ds.green - ds.nir) / (ds.green + ds.nir)
            elif analysis_type=="evi":
                res= 2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
                res=xr.where(~np.isfinite(res),0.0,res)
            elif analysis_type=="graph":
                ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)
                evi = 2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))

                # Create forest masks based on NDVI and EVI thresholds
                forest_mask = np.where((ndvi > 0.5) & (evi > 0.2), 1, 0)

                # Calculate the area of each pixel
                pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4])
                print('pixel_area', pixel_area)

                data = [['day', 'month', 'year', 'forest', 'total']]

                for i in range(forest_mask.shape[0]):
                    data_time = str(ndvi.time[i].values).split("T")[0]
                    new_data_time = data_time.split("-")

                    # Calculate the forest cover area for each forest type
                    forest_cover_area = np.sum(forest_mask[i]) * pixel_area

                    original_array = np.where(ndvi > -10, 1, 0)
                    original = np.sum(original_array[i]) * pixel_area
                    
                    data.append([new_data_time[2], new_data_time[1], new_data_time[0],
                                forest_cover_area, original])
                
                df = pd.DataFrame(data[1:], columns=data[0])
                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')

                grouped_df = df.groupby(['year', 'month'])

                # Step 3: Calculate the mean of 'forest_field' for each group
                mean_forest_field = grouped_df['forest'].mean()

                # Step 4: Optional - Reset the index of the resulting DataFrame
                mean_forest_field = mean_forest_field.reset_index()
                print(mean_forest_field)

                df = mean_forest_field

                X = df[["year", "month"]]
                y = df["forest"]

                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
                rf_regressor.fit(X, y)
                y_pred = rf_regressor.predict(X)
                print(df, y_pred)

                df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')
                X["year-month"] = X["year"].astype('str') + "-" + X["month"].astype('str')

                print("year-month done")

                plot_data = [
                    go.Scatter(
                        x = df['year-month'],
                        y = df['forest']/1000000,
                        name = "Forest Actual"
                    ),
                    go.Scatter(
                        x = df['year-month'],
                        y = y_pred/1000000,
                        name = "Forest Predicted"
                    )
                ]

                print("Plot plotted")

                plot_layout = go.Layout(
                    title='Forest Cover'
                )
                fig = go.Figure(data=plot_data, layout=plot_layout)

                fig.update_layout(
                    xaxis_title="Year-Month",
                    yaxis_title="Forest Area (sq.km.)"
                )
                # Convert plot to JSON
                plot_json = pio.to_json(fig)

                area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
                print(area_name)

                return jsonify({"plot": plot_json, "type": "Random Forest Analysis", "area_name": area_name})
            else:
                return jsonify({"error": "Invalid type"})

            res_start = res.sel(time=slice(time_range[0], time_range[1])).min(dim='time')
            res_end = res.sel(time=slice(time_range[0], time_range[1])).max(dim='time')
            res_diff = res_end - res_start
            print(time_range)

            if analysis_type=="ndvi":
                title = 'Vegetation'
                # cmap = 'RdYlBu'
                cmap = 'Greens'
            elif analysis_type=="ndwi":
                title = 'Water'
                # cmap = 'RdBu'
                cmap="Blues"
            elif analysis_type=="evi":
                title = "EVI"
                cmap='viridis'

            sub_res = res.isel(time=[0, -1])
            mean_res = res.mean(dim=['latitude', 'longitude'], skipna=True)
            mean_res_rounded = list(map(lambda x: round(x, 4), mean_res.values.tolist()))
            labels = list(map(lambda x: x.split('T')[0], [i for i in np.datetime_as_string(res.time.values).tolist()])) 

            plot = sub_res.plot(col='time', col_wrap=2)
            for ax, time in zip(plot.axes.flat, res.time.values):
                ax.set_title(str(time).split('T')[0])

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)

            # plt.figure(figsize=(10, 6))
            # gs = gridspec.GridSpec(1,2)

            # plt.subplot(gs[0,0])
            # plt.imshow(res_start, cmap=cmap, vmin=-1, vmax=1)
            # plt.title(title+' '+data['fromdate'][:4])

            # plt.subplot(gs[0,1])
            # plt.imshow(res_end, cmap=cmap, vmin=-1, vmax=1)
            # plt.title(title+' '+data['todate'][:4])

            

            # plt.colorbar()

            now = datetime.now()
            timestamp = now.strftime("%d/%m/%Y at %I:%M:%S %p")
            plt.xlabel(timestamp)
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
            plt.clf()

            area_name = get_area_name(np.mean(study_area_lat), np.mean(study_area_lon))
            print(area_name)
            
            return jsonify({"plot_url": plot_data,  "data": str(dict(request.form)), "coordinates": coordinates,"area_name":area_name,"type": analysis_type, "mean_res_rounded": mean_res_rounded, "labels": labels})
        except Exception as e:
            return jsonify({"error": e})
    return jsonify({"error": "Invalid method: "+request.method})

def get_area_name(latitude, longitude):
    geolocator = Nominatim(user_agent='my-app')  # Initialize the geocoder
    location = geolocator.reverse((latitude, longitude))  # Reverse geocode the coordinates
    if location is not None:
        address_components = location.raw['address']
        city_name = address_components.get('city', '')
        if not city_name:
            city_name = address_components.get('town', '')
        if not city_name:
            city_name = address_components.get('village', '')
        return city_name
    else:
        return "City name not found"

@app.route('/datasets', methods=['GET'])
def datasets():
    dc = datacube.Datacube(app='datacube-example')
    product_name = ['s2a_sen2cor_granule']

    p = []

    for j in product_name:
        datasets = dc.find_datasets(product=j)
        d = []
        if len(datasets) == 0:
            print('No datasets found for the specified query parameters.')
        else:
            for i in datasets:
                ds_loc = i.metadata_doc['geometry']['coordinates']
                d.append(ds_loc)
        unique_list = [x for i, x in enumerate(d) if x not in d[:i]]
        p+=unique_list
    unique_list = [x for i, x in enumerate(p) if x not in p[:i]]
    print(unique_list)
    return jsonify({'coordinates': unique_list})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')