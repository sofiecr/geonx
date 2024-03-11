import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import MultiPoint, Point, LineString
from shapely import segmentize
from shapely.ops import split



def decompose_edge(edge, edge_attr, max_length, next_node_ID):
    '''
    Decompose an edge into edges with max_length.

    Parameters
    ----------
    edge : row of a pandas GeoDataframe containing a LineString as geometry
    edge_attr: list
        edge attributes to copy
    max_length: int
        the maximum length of the resulting edge
    next_node_ID: int
        the starting ID of the new nodes

    Returns
    -------
    new_nodes_dict, new_edges_dict

    new_nodes_dict: a dictionary of nodes to add to a nodes geodataframe
    new_edges_dict: a dictionary of edges to add to an edge geodataframe
    '''
    #print(next_node_ID)
    #1. split edges in segments of max_length
    line = edge.geometry
    u, v, key = edge.Index #u = startnode, v = endnode
    old_osmid = edge.osmid

    segmented_line = segmentize(line, max_length)
    pps = MultiPoint([Point(segmented_line.coords[i]) for i in range(1, len(segmented_line.coords)-1)])
    new_lines = split(segmented_line, pps)

    #2. create new nodes
    n_new_nodes = len(pps.geoms)
    node_ids = [u] + [next_node_ID + i for i in range(n_new_nodes)] + [v]

    #get new node ids (for use in edges)
    dict_new_lines_ids = {i: (node_ids[i], node_ids[i+1], key) for i in range(len(node_ids)) if i < len(node_ids)-1}
    n_new_lines = len(dict_new_lines_ids)

    #print(dict_new_lines_ids)
    #create new nodes dict
    new_nodes_dict = [{'osmid': node_ids[i + 1], 'geometry': pps.geoms[i]} for i in range(n_new_nodes)]
    #print(f"lengte nodes dict in decompose edge: {len(new_nodes_dict)}")
    #new_nodes_dict = {'osmid': node_ids[1:-1],
    #                'geometry': [p for p in pps.geoms]}
    
    #3. create new edges
    #get edge attributes from original edge
    dict_edge_attr = {attr: getattr(edge, attr) for attr in edge_attr}


    #create new edges
    new_edges_dict = [{'u': dict_new_lines_ids[i][0], 
                    'v': dict_new_lines_ids[i][1], 
                    'key': dict_new_lines_ids[i][2], 
                    'geometry': new_lines.geoms[i],
                    'length': new_lines.geoms[i].length,
                    'old_osmid': old_osmid} for i in range(n_new_lines)]

    #add edge attributes to each new edge
    for new_line_dict in new_edges_dict:
        new_line_dict.update(dict_edge_attr)

    return new_nodes_dict, new_edges_dict

def decompose_network(gdf_nodes, gdf_edges, max_length, edge_attr=None):
    '''
    Decomposes a network into edges with max_length

    Parameters
    ----------
    gdf_nodes: geopandas.GeoDataFrame
        containing the nodes
    gdf_edges: geopandas.GeoDataFrame
        containing the edges as LineStrings
    max_length: int
        the maximum length of the resulting edges
    edge_attr (optional): list
        edge attributes to copy
        if None, take all attributes from the original edges, apart from geometry, osmid and length
    Returns
    -------
    shapely.geometry.Point
    '''

    if edge_attr == None:
        edge_attr = list(gdf_edges.columns)
        edge_attr.remove('geometry') # don't include the old geometry
        edge_attr.remove('osmid')
        edge_attr.remove('length')

    edges_to_split = gdf_edges[gdf_edges['length']>max_length]

    lst_edges_to_drop = []
    new_edges_lst = []
    new_nodes_lst = []
    
    next_node_ID = max(gdf_nodes.index.to_list()) + 1

    for edge in edges_to_split.itertuples():
            
        u, v, key = edge.Index
        
        #get new nodes and edges
        new_nodes_dict, new_edges_dict = decompose_edge(edge, edge_attr, max_length, next_node_ID)
        next_node_ID += len(new_nodes_dict)
        
        #remove original edge
        lst_edges_to_drop.append((u, v, key))
            
        #add to list of new edges
        new_edges_lst.extend(new_edges_dict)
        
        #add to list new nodes
        new_nodes_lst.extend(new_nodes_dict)


    #create geo dataframe of new edges and add to existing edges
    new_edges_gdf = gpd.GeoDataFrame(new_edges_lst, crs=gdf_edges.crs).set_index(['u', 'v', 'key'])
    gdf_edges = pd.concat([gdf_edges, new_edges_gdf])
    
    #create dataframe of new nodes and add to existing nodes
    new_nodes_gdf = gpd.GeoDataFrame(new_nodes_lst, crs=gdf_nodes.crs).set_index('osmid')
    gdf_nodes = pd.concat([gdf_nodes, new_nodes_gdf])

    #drop edges that were decomposed
    gdf_edges = gdf_edges.drop(lst_edges_to_drop)

    #fill x and y value in nodes_df
    gdf_nodes['x'].fillna(gdf_nodes.geometry.x, inplace=True)
    gdf_nodes['y'].fillna(gdf_nodes.geometry.y, inplace=True)

    #to lat-lon
    gdf_nodes_wgs84 = gdf_nodes.to_crs(4326)
    gdf_nodes['lon'].fillna(gdf_nodes_wgs84.geometry.x, inplace=True)
    gdf_nodes['lat'].fillna(gdf_nodes_wgs84.geometry.y, inplace=True)

    return gdf_nodes, gdf_edges

def project_point_on_edge(point, gdf_edges, name_col, streetname=None):
    '''
    Project a point on the nearest edge (maximum distance of 300).
    If a streetname is defined then fuzzy match (score > 70) on streetname and then find the nearest edge

    Parameters
    ----------
    point : shapely.geometry.Point
    gdf_edges: geopandas.GeoDataFrame
        containing the edges as LineStrings
    name_col: str
        the column containing the streetname in gdf_edges (lower case and stripped)
    streetname: str
        the original streetname of the edge (lower case and stripped)

    Returns
    -------
    projected_point: shapely.geometry.Point
        point on the nearest edge (with optional streetname match)
    result.distance_to_edge: float
        distance from the original point to the projected_point
    (u, v, key): tuple
        index of the nearest edge in gdf_edges (with optional streetname match)
    '''
    
    from thefuzz import process

    #helper functions
    def check_string(list_of_strings, search_term):
        return any(search_term in string for string in list_of_strings) 

    def check_string_fuzzy_match(col_array, search_term, min_score):
        fuzzy_matches = [el for el, score in process.extractBests(search_term, col_array, score_cutoff=min_score, limit=15)]
        result = set().union(*map(set, fuzzy_matches))
        if result:
            return True
        else:
            return False
        
    #start of the function
    try:
        gdf_point = gpd.GeoDataFrame(geometry=[point], crs=gdf_edges.crs)
        gdf_edges_subset = gdf_edges[['geometry', name_col]]

        if streetname:
            #filter gdf_edges on streetname
            streetnames = gdf_edges_subset[name_col].values
            vectorized_check_string = np.vectorize(check_string)
            result_array = vectorized_check_string(streetnames, streetname)
            gdf_edges_subset=gdf_edges_subset[result_array]

            if gdf_edges_subset.empty: 
                #no matches => apply fuzzy matching
                vectorized_fuzzy_match = np.vectorize(check_string_fuzzy_match)
                result_array = vectorized_fuzzy_match(streetnames, streetname, min_score=70)
                gdf_edges_subset=gdf_edges[result_array]
                if gdf_edges_subset.empty:
                    return [None, None, 'There were no edges with (fuzzy) matching streetname']
        
        gdf_join = gpd.sjoin_nearest(gdf_point, gdf_edges_subset, how='left', distance_col='distance_to_edge', max_distance=300)
        
        if len(gdf_join) > 1: #more matches
            if len(gdf_join['distance_to_edge'].unique())==1: #but same distance (e.g. same street in both directions)
                #take the first one, doesn't matter
                gdf_join = gdf_join.iloc[[0]]
                
        if len(gdf_join) == 1:
            result = gdf_join.iloc[0]
            u, v, key = result.index_right0, result.index_right1, result.index_right2
            edge_line = gdf_edges.loc[(u, v, key)].geometry
            projected_point = edge_line.interpolate(edge_line.project(point))
    
            return [projected_point, result.distance_to_edge, (u, v, key)]
        else:
            return [None, None, 'No results in gdf_join']

    except:
        return [None, None, 'Error']

def add_poi_to_network(point, gdf_nodes, gdf_edges, point_ID=None):
    '''
    Projects a point on the nearest edge and adds this as a node to the network.
    The nearest edge is split into two segments (with the same attributes as the original edge)
    
    Parameters
    ----------
    point : shapely.geometry.Point
    gdf_edges: geopandas.GeoDataFrame
        containing the edges as LineStrings
    gdf_nodes: geopandas.GeoDataFrame
        containing the nodes as Point
    point_ID: str or int
        ID to use in error message
    Returns
    -------
    (gdf_nodes, gdf_edges): tuple
    
    gdf_nodes: geopandas.GeoDataFrame
        the original gdf_nodes with a node added
    gdf_edges: geopandas.GeoDataFrame
        the original gdf_edges with the original edge replaced by two edges split on the added node
    '''
    
    #1. find nearest edge
    gdf_point = gpd.GeoDataFrame(geometry=[point], crs=gdf_edges.crs)
    gdf_join = gpd.sjoin_nearest(gdf_point, gdf_edges, how='left', distance_col='distance_to_edge', max_distance=300)

    if len(gdf_join) > 1: #more matches
        if len(gdf_join['distance_to_edge'].unique())==1: #but same distance (e.g. same street in both directions)
            #take the first one, doesn't matter
            gdf_join = gdf_join.iloc[[0]]

    if len(gdf_join) == 1:
        gdf_edge = gdf_edges.loc[(gdf_join.index_right0, gdf_join.index_right1, gdf_join.index_right2)]
        edge = gdf_edge.iloc[[0]]
        line = edge.geometry.iloc[0]
        (u, v, key) = edge.index[0]
        old_osmid = edge.osmid.iloc[0]
        pp = line.interpolate(line.project(point))
        
        startnode = gdf_nodes.loc[gdf_join.index_right0].geometry.iloc[0] #startnode line
        endnode = gdf_nodes.loc[gdf_join.index_right1].geometry.iloc[0] #endnode line

        if startnode.distance(pp) < 5 and endnode.distance(pp) < 5: #only split if distance to existing node > 5m
            return gdf_nodes, gdf_edges
        
        #2. split edge on point
        new_lines = split(line, pp.buffer(1.0e-6))

        #because of the buffer, 3 lines are returned (the middle one being very small)
        #create 2 lines from these 3 lines and check if it equals the original length
        if len(new_lines.geoms) == 1:
            if point_ID:
                print(f'Problem splitting line by point {point_ID}')
            else:
                print('Problem splitting line by point')
        elif len(new_lines.geoms) == 3:
            line1 = LineString([Point(new_lines.geoms[0].coords[0]), Point(new_lines.geoms[1].coords[0])])
            line2 = LineString([Point(new_lines.geoms[1].coords[0]), Point(new_lines.geoms[2].coords[1])])
            assert line1.length + line2.length == line.length
        elif len(new_lines.geoms) == 2:
            line1 = new_lines.geoms[0]
            line2 = new_lines.geoms[1]
            
        #3. create new node
        next_node_ID = max(gdf_nodes.index.to_list()) + 1
        new_nodes_dict = [{'osmid': next_node_ID, 'geometry': pp}]
        
        #3. create new edges
        #get edge attributes from original edge
        edge_attr = list(gdf_edges.columns)
        edge_attr.remove('geometry') # don't include the old geometry
        edge_attr.remove('osmid')
        edge_attr.remove('length')

        #dict_edge_attr = {attr: getattr(edge, attr) for attr in edge_attr}
        dict_edge_attr = {attr: edge[attr].iloc[0] for attr in edge_attr}

        #create new edges
        new_edges_dict = [{'u': u, 'v': next_node_ID, 'key': key, 'geometry': line1, 'length': line1.length, 'old_osmid': old_osmid},
                        {'u': next_node_ID, 'v': v, 'key': key, 'geometry': line2, 'length': line2.length, 'old_osmid': old_osmid}]

        #add edge attributes to each new edge
        for new_line_dict in new_edges_dict:
            new_line_dict.update(dict_edge_attr)

        #create geo dataframe of new edges and add to existing edges
        new_edges_gdf = gpd.GeoDataFrame(new_edges_dict, crs=gdf_edges.crs).set_index(['u', 'v', 'key'])
        gdf_edges = pd.concat([gdf_edges, new_edges_gdf])
        
        #create dataframe of new nodes and add to existing nodes
        new_nodes_gdf = gpd.GeoDataFrame(new_nodes_dict, crs=gdf_nodes.crs).set_index('osmid')
        gdf_nodes = pd.concat([gdf_nodes, new_nodes_gdf])

        #drop edge that was decomposed
        gdf_edges = gdf_edges.drop([(u, v, key)])

        #fill x and y value in nodes_df
        gdf_nodes['x'].fillna(gdf_nodes.geometry.x, inplace=True)
        gdf_nodes['y'].fillna(gdf_nodes.geometry.y, inplace=True)

        #to lat-lon
        gdf_nodes_wgs84 = gdf_nodes.to_crs(4326)
        gdf_nodes['lon'].fillna(gdf_nodes_wgs84.geometry.x, inplace=True)
        gdf_nodes['lat'].fillna(gdf_nodes_wgs84.geometry.y, inplace=True)
    else:
        if point_ID:
            print(f'point {point_ID} could not be added to the network')
        else:
            print(f'point ({point.x}, {point.y}) could not be added to the network')

    return gdf_nodes, gdf_edges