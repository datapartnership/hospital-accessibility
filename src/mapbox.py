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

mapbox_tokens = []
mapbox_tokens.append(os.environ.get("MAPBOX_TOKEN", "NO_TOKEN"))

    
def get_travel_times_mapbox(origins, 
                            destinations, 
                            mode, 
                            d_name, 
                            dest_id_col=False,
                            n_keep = 2,
                            num_retries = 2, 
                            starting_token_index = 0,
                            use_pickle = False,
                            do_pickle_result=True,
                            pickle_region_name = "",
                            batch_limit=None):
    """
    
    """
    
    # Immediately return pickled data if requested
    pickle_name = f"../data/interim/mb_origins_{pickle_region_name}_{d_name}_{mode}"
    if use_pickle == True:
        return unpickle_data(picke_name)
    

    #tries = 0
    token_index = starting_token_index
    #while tries <= num_retries:
    origins = mapbox_matrix_API(
        token = mapbox_tokens[token_index],
        origins = origins,
        destinations = destinations,
        mode = mode,
        d_name = d_name,
        dest_id_col = dest_id_col,
        n_keep = n_keep,
        batch_limit = batch_limit
    )
    origins[f"closest_{d_name}_geom"] = gpd.points_from_xy(
        origins.loc[:, f"closest_{d_name}_geom_lon_x"],
        origins.loc[:, f"closest_{d_name}_geom_lat_y"]
    )
    origins["mb_snapped_dest_geom"] = gpd.points_from_xy(
        origins.loc[:, "mb_snapped_dest_lon_x"],
        origins.loc[:, "mb_snapped_dest_lat_y"]
    )
    origins["mb_snapped_src_geom"] = gpd.points_from_xy(
        origins.loc[:, "mb_snapped_src_lon_x"],
        origins.loc[:, "mb_snapped_src_lat_y"]
    )
    
     # Pickle generated origins if requested
    if do_pickle_result == True:
        pickle_data(origins, pickle_name)
     #   tries += 1

    return origins





def mapbox_matrix_API(token, 
                      origins, 
                      destinations, 
                      mode='driving', 
                      d_name='poi',
                      dest_id_col='osmid',
                      n_keep=3, 
                      batch_limit=None):
    """
    Given a geopandas set of origins and destinations, return the origins with extra columns
    with the closest destination in minutes given the mode of transportation for each origin.
    
    Also returns the snap distance to the origin (geodetic distance from origin point to closest road)
    Keywords:
    do_all [False]: By default avoid repeating work that has been done.
    
    """

    MAPBOX_TOKEN = token
    osrm_server="https://api.mapbox.com/directions-matrix/v1"
    modes=['driving-traffic', 'driving', 'cycling', 'walking']
    if mode not in modes:
        raise ValueError("Mode should be one of [driving-traffic, driving, cycling, walking]")

    url = f"{osrm_server}/mapbox/{mode}"

    max_coordinates = 24
    
    if mode == "driving-traffic":
        max_coordinates = 10


    # Limit Coordinates Total = 25
    # Since limit is 25, that means that # sources + # destinations = 24

    batch_size = int(np.floor(max_coordinates / (n_keep + 1)))

    # Append proper columns
    for c in [
        f"hrs_to_{d_name}",
        f"mins_to_{d_name}",
        f"dist_to_{d_name}",
        f"closest_{d_name}_id",
        f"closest_{d_name}_name",
        f"closest_{d_name}_geom_lon_x",
        f"closest_{d_name}_geom_lat_y",
        f"closest_{d_name}_geodetic_dist",
        "mb_snapped_dest_name",
        "mb_snapped_dest_dist",
        "mb_snapped_dest_lon_x",
        "mb_snapped_dest_lat_y",
        "mb_snapped_src_name",
        "mb_snapped_src_dist",
        "mb_snapped_src_lon_x",
        "mb_snapped_src_lat_y"

    ]:
        origins[c] = -1


    

    # Out of the unprocessed origins, get the next batch to process
    # Get origins where time to hospital is -1 or unprocessed
    queued_origins = origins.loc[origins[f"hrs_to_{d_name}"] == -1, :]
    queued_origins_size = queued_origins.shape[0]
    
    for iteration in np.arange(queued_origins_size / batch_size):
        if ((batch_limit is not None) and int(iteration) >= batch_limit):
            break

        queued_origins = origins.loc[origins[f"hrs_to_{d_name}"] == -1, :]
        print(f"""
        Process batch # {int(iteration)} 
        Remaining: {queued_origins.shape[0]}
        """)
        #origins_to_process = queued_origins.iloc[int(batch_size*iteration):].head(batch_size).copy()
        

        origins_to_process = queued_origins.head(batch_size).copy()
    

        relevant_destinations = n_closest_geodetic(origins_to_process, destinations, n_keep)

        # Safety check
        if (len(origins_to_process) + len(relevant_destinations)) > max_coordinates:
            raise ValueError("Over limit for Mapbox API")

        origins_url = origins_to_process[['geometry']].copy().reset_index(drop=False)
        dest_url = relevant_destinations[['geometry']].copy().reset_index(drop=False)
        origins_url['type'] = 'origin'
        dest_url['type'] = 'dest'

        od = pd.concat([origins_url, dest_url]).reset_index(drop=True)

        origins_coords = ";".join([",".join([str(row.centroid.x),str(row.centroid.y)]) for row in od.loc[od['type'] =='origin','geometry']])
        relevant_dest_coords=";".join([",".join([str(row.centroid.x),str(row.centroid.y)]) for row in od.loc[od['type'] == 'dest', 'geometry']])


        origin_coords_indices = ';'.join([str(x) for x in od.loc[od['type'] == 'origin', :].index.tolist()])
        dest_coords_indices = ';'.join([str(x) for x in od.loc[od['type'] == 'dest', :].index.tolist()])


        full_url = f"{url}/{origins_coords};{relevant_dest_coords}.json?sources={origin_coords_indices}&destinations={dest_coords_indices}&annotations=distance,duration&access_token={MAPBOX_TOKEN}"


        response = requests.get(full_url)
        response.raise_for_status()


        response = json.loads(response.text)
    
    
        durations = response['durations']
        distances = response['distances']
        mb_dests = response['destinations']
        sources = response['sources']


        for ix, dur_set in enumerate(distances):
            if len(dur_set) != len(relevant_destinations):
                raise ValueError("Incorrect response from Mapbox")
                
            
            #Look up what the index is of this origin
            origin_ix = int(od.loc[:, "index"][ix])
            #print(origin_ix)
            
            
            

            # Clean the dataset for any non-routes, keeping index
            dur_set = list(map(lambda x: x if x else 99999999999999, dur_set))

            travel_time_to_closest_dest = min(dur_set)
            closest_dest_ix = np.argmin(dur_set)
            distance_to_closest_dest = distances[ix][closest_dest_ix]
            closest_dest_osm = relevant_destinations.iloc[closest_dest_ix]
            closest_dest_mb = mb_dests[closest_dest_ix]
            source_mb = sources[ix]




            # Closest OSM Destination
            origins_to_process.loc[origin_ix,  f'hrs_to_{d_name}'] = ((travel_time_to_closest_dest / 60) / 60)
            origins_to_process.loc[origin_ix,  f'mins_to_{d_name}'] = (travel_time_to_closest_dest / 60)
            origins_to_process.loc[origin_ix, f'dist_to_{d_name}'] = distance_to_closest_dest

            origins_to_process.loc[origin_ix, f"closest_{d_name}_id"] = closest_dest_osm[dest_id_col]
            origins_to_process.loc[origin_ix, f"closest_{d_name}_name"] = closest_dest_osm['name']
            origins_to_process.loc[origin_ix, f"closest_{d_name}_geodetic_dist"] = closest_dest_osm['distance_to_or']
            origins_to_process.loc[origin_ix, f"closest_{d_name}_geom_lon_x"] = closest_dest_osm['geometry'].x
            origins_to_process.loc[origin_ix, f"closest_{d_name}_geom_lat_y"] = closest_dest_osm['geometry'].y


            # Closest MB Destination
            origins_to_process.loc[origin_ix, f"mb_snapped_dest_name"] = closest_dest_mb['name']
            origins_to_process.loc[origin_ix, f"mb_snapped_dest_dist"] = closest_dest_mb['distance']
            origins_to_process.loc[origin_ix, f"mb_snapped_dest_lon_x"] = closest_dest_mb['location'][0]
            origins_to_process.loc[origin_ix, f"mb_snapped_dest_lat_y"] = closest_dest_mb['location'][1]

            # Closest Source
            origins_to_process.loc[origin_ix, f"mb_snapped_src_name"] = source_mb['name']
            origins_to_process.loc[origin_ix, f"mb_snapped_src_dist"] = source_mb['distance']
            origins_to_process.loc[origin_ix, f"mb_snapped_src_lon_x"] = source_mb['location'][0]
            origins_to_process.loc[origin_ix, f"mb_snapped_src_lat_y"] = source_mb['location'][1]




            
            #dest_geom = relevant_destinations.loc[:, "geometry"][closest_dest_ix]
            #origins_to_process.loc[origin_ix, f"closest_{d_name}_geometry"] = dest_geom


        # Get the new stuff into origins
        origins.loc[origins_to_process.index, :] = origins_to_process.loc[:,:].copy()

    
    queued_origins = origins.loc[origins[f"hrs_to_{d_name}"] == -1, :]
    queued_origins_size = queued_origins.shape[0]
    
    print(f"There are still {queued_origins_size} unprocessed origins")
    return origins


def mapbox_analysis(mode,origins,destinations):
    
    """
    Given origins, destinations and filtered destinations conducts analysis on travel distane for origin-destination pairs 
    """
    
        
    # Temp
    #failed_data = origins.loc[origins.t_hospital == -1, :]
    #origins = origins.loc[origins.t_hospital != -1, :].copy()
    #failed_data = failed_data.reset_index(drop=True)
    #failed_data = failed_data.loc[failed_data.index != 1, :]
    #origins = pd.concat([origins, failed_data]).reset_index(drop = True)
    #origins.shape
    
    #Data Formatting 
    #origins.to_file("./data/origins.geojson", driver="GeoJSON")
    #destinations.to_file("./data/dest.geojson", driver="GeoJSON")
    #o = gpd.read_file("./data/origins.geojson")
    #h = gpd.read_file("./data/dest.geojson")
    #o.head()
    
    o = origins.copy()
    h = destinations.copy()
    
    #Plotting 
    #%matplotlib inline
    plt.figure()
    o['hrs_to_hosp_or_clinic'].plot.hist(mode, alpha=0.7,bins=1000,cumulative=False,density=False,log=False,logx=False,weights=o['tot_pop'])
    plt.xlim((0,2))
    plt.title(mode)
    plt.ylabel('Population 1e7')
    plt.xlabel('Distance to closest: Hospital')
    plt.show()
