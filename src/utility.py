
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



def plot_df(df, geo_col = 'geometry'):
    """
    Function plots dataframe of interest's geometry on base map
    df - dataframe
    """
    dfp = gpd.GeoDataFrame(df, geometry = df[geo_col])
    dfp = dfp.set_crs(df.crs)
    dfp = dfp.to_crs("EPSG:3857")
    ax = dfp.plot(figsize=(10, 10), alpha=0.8)
    ctx.add_basemap(ax)
    ax.set_axis_off()



def plot_df_single(df, geo_col = 'geometry', hover_cols = None, alpha=0.5, color=None):
    """
    Function maps data frame of interest on an interactive map with zooming features and hover info
    df-dataframe
    geo_col-geometry type to be plotted 
    hover_cols=column data that should appear when hovering over data point
    """
    dfp = gpd.GeoDataFrame(df, geometry = df[geo_col])
    dfp = dfp.set_crs(df.crs)
    if color == None:
        colors = ['green', 'blue', 'orange', 'red', 'yellow', 'pink', 'cyan', 'magenta']
        color = colors[random.randint(0, len(colors) - 1)]
    plot =  dfp.hvplot(
        geo=True, 
        frame_width=500, 
        frame_height=500,
        alpha=alpha, color=color, hover_cols=hover_cols,legend='top')

    tiles = gv.tile_sources.OSM()
    return tiles * plot
    
    


def pickle_data(df, pickle_path):
     with open (pickle_path, "wb") as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Pickled data")
    
    
def unpickle_data(pickle_path):
    with open (pickle_path, "rb") as handle:
            data = pickle.load(handle)
            print(f"Unpickled data at {pickle_path}: \n Rows: {data.shape};")
            return data


