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
from utility import *
from raster_ops import *
from mapbox import *


  
def get_origins(places, population_file, window_size = 50, use_pickle = False, do_pickle_result = True, pickle_region_name = ""):
    
    """
    Function extracts origins points from raster population map-based origin data 
    places - string of region of interest, or a dataframe of administrate boundary polygon
    population_file - raster file
    window_size - final size of each population grid that is wanted 
    """
    
    
    # Immediately return pickled data if requested
    if use_pickle == True:
        with open (f"../data/interim/origins_{pickle_region_name}.pickle", "rb") as handle:
            origins = pickle.load(handle)
        print(f"Origins:{origins.shape};")
        return origins

    #Scan the raster map with big windows
    origins=pd.DataFrame()
    window=window_size * 2
    with rasterio.open(population_file) as src:
        for left_x in np.arange(0,src.width,window):
            for top_y in np.arange(0,src.height,window):
                out=get_pop(population_file,left_x,top_y,window,plot=False)
                if out != {}:
                    origins=origins.append([out])
            print("%i/%i\r"%(left_x,src.width),end="")

    #Do a splitting pass. Run this cell several times (we run four times),
    # until you have a balance of small window and not too big rois
    #run this cell as many times as you want to split the windows

    #for i in range(0,4):
    #for pass_num in range(0,split_passes):
    # Split pass start
    regions_need_splitting = origins[origins['split'] == True]

    print(f"{len(regions_need_splitting)} regions need splitting")

    olen=len(origins)

    for i in np.arange(olen):
        print("%i/%i\r"%(i+1,olen),end="")
        if origins.iloc[i,origins.columns.get_loc('split')] == True:
            origins.iloc[i,origins.columns.get_loc('split')]='done'
            s = split(population_file,origins.iloc[i])
            origins=origins.append(s,sort=False)
    print("done.")
    print("We now have %i regions of min size %i, %i will be split in next round"%\
          (len(origins),origins['window'].min(),len(origins[origins['split']==True])))
    
    origins=origins[origins['tot_pop']>0]
    origins=origins[origins['split']!='done']
    print("We have %i regions of size %i, %i with population >0"%
          (len(origins),min(origins['window']),len(origins[origins['tot_pop']>0])))
    # Split pass end
    
    # Set the geometry of the generated polygons as points - that are grid centroids.
    origins=gpd.GeoDataFrame(origins,
                         crs='epsg:4326', 
                         geometry=[Point(xy) for xy in zip(origins['center_lon'], origins['center_lat'])]
                        )
    
    # Create a separate geometry column that contains the grid geometry
    origins['geo_grid']=origins.apply(
        lambda r: box(r.left_lon, r.bottom_lat, r.right_lon, r.top_lat, ccw=False),
        axis=1
    )
    
    
    col_to_keep = origins.columns

    # Filter Origins with administrative boundary 
    
    if (isinstance(places,pd.DataFrame)==True):
        bounds = places
        
    else:
        bounds = ox.boundaries.gdf_from_places(places)
    
    # Don't clip to full bounds, just bbox. That's why this is commented
    tr_origins = gpd.sjoin(origins, bounds, how="inner", op="intersects")
    tr_origins = tr_origins[col_to_keep].reset_index(drop=True)

    
    #Outputting Origins 
    print(f"All origins:{origins.shape}; Relevant Origins:{tr_origins.shape}")
    
    # Pickle generated origins if requested
    if do_pickle_result == True:
        with open ("../data/interim/origins_{pickle_region_name}.pickle", "wb") as handle:
            pickle.dump(tr_origins, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("Pickled origins")
    
    return(tr_origins)



def origins_snap_osm(origins, net_pickle_path):
    """
    Function snaps origins to nearest road based on road network
    origins - dataframe of population origin points
    net_pickle_path - road network
    """
    print("Reading pickle")
    G = nx.read_gpickle(net_pickle_path)

    print("Snapping Origins")
    origins = gn.pandana_snap_c(G, 
                                point_gdf = origins,
                                source_crs = 'epsg:4326',
                                target_crs = 'epsg:4326',
                                add_dist_to_node_col = True,
                                time_it = True)


    origins = origins.drop(['x', 'y'], axis=1)

    print("Converting NNs to Points")
    def get_geo_from_node(NN):
        node = G.nodes[NN]
        return gpd.points_from_xy([node['x']], [node['y']])[0]

    origins['center_geom'] = origins['geometry'].copy()

    origins['geometry'] = origins['NN'].apply(get_geo_from_node)
    origins['geometry'] = origins['geometry'].set_crs('epsg:4326')

    return origins



def get_destinations(places, tags):
    """
    Function extract POI data from OSM
    places - string of region of interest, or a dataframe of administrate boundary polygon
    tags - amenity tags for destinations, must be queryable in OSM
    """
    #For places innputs tht are a data frame 
    if isinstance(places, pd.DataFrame)==True :
        bounds = places.reset_index(drop=True)
        boundaries=bounds.loc[0]['geometry']
        df = ox.pois_from_polygon(boundaries, tags)
   
    #For places inputs that are a string   
    else:
        destinations_list = []
        for p in places:
            destinations_list.append(ox.pois_from_place(p, tags))
        df = pd.concat(destinations_list)
    
    #Formatting dataframe
    df = gpd.GeoDataFrame(df[["osmid", "amenity", "name", "source"]], geometry=df['geometry'])
    df = df.set_crs("EPSG:4326")
    
    # Convert Polygons to centroid
    df.loc[df.geometry.geom_type != 'Point', "geometry"] = df.loc[df.geometry.geom_type != 'Point', "geometry"].centroid
    
    #Making sure we have no Amenitities of None Type
    df = df.loc[df.amenity.isin(tags['amenity']), :].reset_index(drop=True).copy()
    return df

def n_closest_geodetic(origins, destinations, n_keep):
    """
    Function takes in origins and destinations and outputs a new destination list
    with n_keep amount of closest destinations to each origin. This helps make fewer calls from Mapbox
    
    origins - data frane
    destinations - data frane
    n_keep = int number of nearby destinations you would like to keep 
    """
    
    destinations = destinations.to_crs("EPSG:4326")
    origins = origins.to_crs("EPSG:4326")
    
    dest_list = []
    
    for i in origins.index:
        origin = origins.loc[i, :]
        dest_o = destinations.copy()
        dest_o['distance_to_or'] = dest_o.distance(origin.geometry)
        dest_o['o_index'] = i
        dest_o = dest_o.sort_values(by='distance_to_or', axis=0, ascending=True).head(n_keep)
        dest_list.append(dest_o)
    
    return pd.concat(dest_list).reset_index(drop=True)
    


def biplot(origins,destinations, mode, t_max,xlim=False):
    """
    Function plos a map and an histogram for the places beyong t_max hours from closest hospital.
    oorigins - data frame with travel time and distance data and population data
    destinations - data frame
    mode - string of travel mode 
    t_max - travel time threshold in hours 
    x_lim - x axis limit for histogram 
    """
    o = origins.copy()
    h = destinations.copy()
    #o_above = o[(o['t_'+o_type]>t_max) & (o['so_'+o_type]<so_max)]
    o_above = o[(o['hrs_to_hosp_or_clinic']>t_max)]
    
    variable="tot_pop"
    vmin,vmax=0,10000
    
    fig = plt.figure(figsize=(12, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 

    ax1=plt.subplot(gs[0])
    ax1.axis("off")
    
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
                       
    fig.colorbar(sm)
    
    o_above['geometry'] = o_above['geo_grid']
    o_above.to_crs('epsg:3857').plot(column=variable, cmap='Reds', linewidth=0.8, ax=ax1,edgecolor='0.8')
    h.to_crs('epsg:3857').plot( alpha=0.8,color='blue',marker=".",markersize=8,ax=ax1)
    
    ctx.add_basemap(ax1)
    ax1.set_title("(Red) population beyond %i h from hospital (Blue)"%t_max)
    ax1.set_axis_off()

    ax2=plt.subplot(gs[1])
    o['hrs_to_hosp_or_clinic'].plot.hist(alpha=0.5,bins=1000,cumulative=True,density=False,log=False,logx=False,weights=o['tot_pop'])
    if len(xlim)>0:
        plt.xlim(xlim)
    ax2.ticklabel_format(style='sci')
    ax2.axvline(t_max,color="red")
    ax2.set_ylabel('People [10s Million]')
    ax2.set_xlabel('Distance to closest hospital or clinic:'+' [h]')
    modestring="%i people (%.2f%%) > %i h "+ mode+ " hospital"
    ax2.set_title(modestring%\
    (o_above['tot_pop'].sum(),o_above['tot_pop'].sum()/o['tot_pop'].sum()*100,t_max))

    #plt.tight_layout()
    plt.show();


