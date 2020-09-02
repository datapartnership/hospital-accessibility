import requests, json, os, pickle
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
import GOSTnets as gn
import networkx as nx



mapbox_tokens = []
mapbox_tokens.append(os.environ.get("MAPBOX_TOKEN", "NO_TOKEN"))


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


def _get_pop(map_file,left_x,top_y,window,plot=False):
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

def _split(map_file,origin,plot=False):
    """
    Split a window row in 4 parts, and return new rows results
    """
    origins=pd.DataFrame()
    
    window=int(origin.window/2)

    for left_x in np.arange(origin.left_x,origin.right_x,window):
        for top_y in np.arange(origin.top_y,origin.bottom_y,window):
            out=_get_pop(map_file,left_x,top_y,window,plot=plot)
            if out != {}:
                origins=origins.append([out])
    return origins

def get_origins(places, population_file, window_size = 50, use_pickle = False, do_pickle_result = True):
    
    """
    Function extracts origins points from raster population map-based origin data 
    places - string of region of interest, or a dataframe of administrate boundary polygon
    population_file - raster file
    window_size - final size of each population grid that is wanted 
    """
    
    
    # Immediately return pickled data if requested
    if use_pickle == True:
        with open ("./data/origins.pickle", "rb") as handle:
            origins = pickle.load(handle)
        print(f"Origins:{origins.shape};")
        return origins

    #Scan the raster map with big windows
    origins=pd.DataFrame()
    window=window_size * 2
    with rasterio.open(population_file) as src:
        for left_x in np.arange(0,src.width,window):
            for top_y in np.arange(0,src.height,window):
                out=_get_pop(population_file,left_x,top_y,window,plot=False)
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
            s = _split(population_file,origins.iloc[i])
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
        with open ("./data/origins.pickle", "wb") as handle:
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
    
    
def get_travel_times_mapbox(origins, 
                            destinations, 
                            mode, 
                            d_name, 
                            dest_id_col=False,
                            n_keep = 2,
                            num_retries = 2, 
                            starting_token_index = 0,
                            use_pickle = False,
                            do_pickle_result =True,
                            batch_limit=None):
    """
    
    """
    
    # Immediately return pickled data if requested
    pickle_name = f"./data/mb_origins_{d_name}_{mode}"
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

        print(full_url)

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
        

def plot_access(origins,destinations,tmax):
    """
    Given origins, with travel time and distance data, and destinations plots an interactive map of population grids 
    that are further than tmax hours away from nearest destination
    
    origins - data frame with travel time and distance data and population data
    destinations - data frame
    tmax - travel time threshold in hours 
    """
    o = origins.copy()
    h = destinations.copy()
    o_above = o[(o['hrs_to_hosp_or_clinic']>tmax)]

    map= plot_df_single(o_above, 'geo_grid', color = 'blue', alpha = 0.8, hover_cols=['tot_pop']) \
        * plot_df_single(h, 'geometry', color='red', alpha = 0.4)
    
    return map

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
