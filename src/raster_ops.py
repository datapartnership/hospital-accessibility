import pandas as pd
import geopandas as gpd
import rasterio
import requests, json, os, pickle
import networkx as nx
import GOSTnets as gn
import matplotlib.pyplot as plt
from matplotlib import gridspec
from time import sleep
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.plot import * 
from rasterio.mask import * 
import numpy as np
from shapely.geometry import Point
from shapely.geometry import box
import contextily as ctx
import osmnx as ox
from fiona.crs import from_epsg
import pycrs
import geoviews as gv
import hvplot.pandas
import random
import utility



def get_pop(map_file,left_x,top_y,window,plot=False):
    """
    get_pop(raster filename,left_x,top_y,window,plot=False)
    
    Given a raster file, and row,cols ranges,
    return the lonlat of the ranges, nancount, and the nunsum
    
    Optionally plot the raster window [False]
    """
    right_x,bottom_y = left_x + window, top_y + window
    
    with rasterio.open(map_file) as src:
        left_lon, top_lat = src.xy(top_y,left_x )
        right_lon, bottom_lat = src.xy(bottom_y, right_x )
        center_lon , center_lat = (right_lon + left_lon)/2., (top_lat+bottom_lat)/2.
                             #Window(col_off, row_off, width, height)
        w = src.read(1, window=Window(left_x, top_y, window, window))
        if plot:
            plot.imshow(w, cmap='pink')
            plt.show()
        nancount=np.count_nonzero(~np.isnan(w))
        count = np.size(w)
        tot_pop=np.nansum(w)
    if count == 0:
        return {} #Out of bounds
    if tot_pop == 0 or window < 1: #Mark the window to furhter split.
        split=False
    else:
        split=True
    out={'window':window,
         'left_x':left_x,
         'right_x':right_x,
         'top_y':top_y,
         'bottom_y':bottom_y,
         'left_lon':left_lon, 
         'top_lat':top_lat, 
         'right_lon':right_lon,
         'bottom_lat':bottom_lat,
         'center_lon':center_lon , 
         'center_lat':center_lat,
         'count': count,
         'nancount':nancount,
         'tot_pop':tot_pop,
         'split': split}
    return out

def split(map_file,origin,plot=False):
    """
    Split a window row in 4 parts, and return new rows results
    """
    origins=pd.DataFrame()
    
    window=int(origin.window/2)

    for left_x in np.arange(origin.left_x,origin.right_x,window):
        for top_y in np.arange(origin.top_y,origin.bottom_y,window):
            out=get_pop(map_file,left_x,top_y,window,plot=plot)
            if out != {}:
                origins=origins.append([out])
    return origins

def mask (population_file, boundary_df, outputfile):
    """
    Clips a raster file based off of the boundary of interest and saves as a tif file by the name outputfile
    population_file - raster file
    boundary_df - data frame of administrative boundary polygon 
    output_file - string name of output fie saving location and name
    """
    data=rasterio.open(population_file)
    
    #generating a bbox using the max and min bounds of our polygon
    bbox = box(
    minx = boundary_df.bounds.loc[0,"minx"],
    miny = boundary_df.bounds.loc[0,"miny"],
    maxx = boundary_df.bounds.loc[0,"maxx"],
    maxy = boundary_df.bounds.loc[0,"maxy"]
    )

    geo=gpd.GeoDataFrame({'geometry':bbox}, index=[0],crs=bounds.crs)
    
    _plot_df(geo)

    geo=[json.loads(geo.to_json())['features'][0]['geometry']]

    #mask and output file
    with rasterio.open(population_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, geo, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    with rasterio.open(outputfile, "w", **out_meta) as dest:
        dest.write(out_image)
