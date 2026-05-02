import numpy as np
import math
import pandas as pd
import geopandas as gpd
import osmnx as ox
import os
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, LineString



class Vector:
    """
    Class for vector in R^3
    """
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

path_beginning = ".."

c = 3e8 / 1e3 # Speed of light
rsrp_bound = -90
epsilon_prioritization = 1e-2
bbox_dtla = (-118.262265,34.041531,-118.250399,34.049742)
bbox_dtr = (-117.386405,33.977891,-117.375376,33.986747)
bbox_manhattan = (-74.008884,40.715082,-73.970089,40.742387)

def euclidean_distance(u: Vector, v: Vector) -> float:
    """
    Return euclidean distance between two vectors in R^3 in m.
    
    :param u: First vector
    :type u: Vector
    :param v: Second vector
    :type v: Vector
    :return: Euclidean distance between u and v
    :rtype: float
    """
    return np.sqrt((u.x-v.x)**2 + (u.y-v.y)**2 + (u.z-v.z)**2)

def euclidean_distance_2d(u: Vector, v: Vector) -> float:
    """
    Return euclidean distance between two vectors in R^2 in m (x and y only)
    
    :param u: First vector
    :type u: Vector
    :param v: Second vector
    :type v: Vector
    :return: Two-dimensional euclidean distance between u and v
    :rtype: float
    """
    return np.sqrt((u.x-v.x)**2 + (u.y-v.y)**2)

def propagation_delay(u: Vector, v: Vector) -> float:
    """
    Return propagation delay in ms between u and v
    
    :param u: First vector
    :type u: Vector
    :param v: Second vector
    :type v: Vector
    :return: Propagation delay in ms
    :rtype: float
    """

    return euclidean_distance(u,v) / c

def path_loss_los(u: Vector, v: Vector, frequency: float) -> float:
    """
    Calculate UMi Line-Of-Sight (LOS) pathloss between gNB at position u and UE at position v
    
    :param u: Position of gNB
    :type u: Vector
    :param v: Position of UE
    :type v: Vector
    :param frequency: Wave frequency
    :type frequency: float
    :return: UMi LOS Pathloss
    :rtype: float
    """
    distance_2d: float = max(euclidean_distance_2d(u,v),10)
    distance_3d: float = max(euclidean_distance(u,v),10)

    base_station_height: float = u.z
    user_equipment_height: float = v.z

    distance_bp_prime: float = 4 * (base_station_height-1) * (user_equipment_height-1) * (frequency*1e9) / c # meters

    if distance_2d >= 0 and distance_2d <= distance_bp_prime:
        return 32.4 + 21 * np.log10(distance_3d) + 20 * np.log10(frequency)
    
    elif distance_2d > distance_bp_prime and distance_2d <= 5_000:
        return 32.4 + 40 * np.log10(distance_3d) + 20*np.log10(frequency) - 9.5 * np.log10((distance_bp_prime)**2 + (base_station_height - user_equipment_height)**2)

    return 0
    
def path_loss_nlos(u: Vector, v: Vector, frequency: float) -> float:
    distance_2d: float = max(10,euclidean_distance_2d(u,v))
    distance_3d: float = max(10,euclidean_distance(u,v))

    if distance_2d < 10:
        return 0

    base_station_height: float = u.z
    user_equipment_height: float = v.z

    path_loss_prime_nlos = 22.4 + 35.3 * np.log10(distance_3d) + 21.3*np.log10(frequency) - 0.3 * (user_equipment_height - 1.5)

    return max(path_loss_prime_nlos, path_loss_los(u,v,frequency))

def shadow_fading_db(is_los: bool, rng=np.random.default_rng()):
    sigma = 4 if is_los else 7.82
    return rng.normal(0, sigma)

def path_loss(u: Vector, v: Vector, frequency: float, gdf, rng=np.random.default_rng) -> float:
    nlos = is_link_blocked(u,v,gdf)
    if not nlos:
        return path_loss_los(u,v,frequency) + shadow_fading_db(True, rng)
    else:
        return path_loss_nlos(u,v,frequency) + shadow_fading_db(False, rng)

def spectral_efficiency(sinr_db):
    if sinr_db >= 20: #256QAM
        return 7.4063
    elif sinr_db >= 15: # High 64QAM
        return 5.5547
    elif sinr_db >= 10: # Mid 64QAM
        return 4.5
    elif sinr_db >= 5: # 16QAM
        return 2.5703
    elif sinr_db >= 0: #QPSK
        return 1.1758
    elif sinr_db >= -5: # Low-rate QPSK
        return 0.4500
    else: # Outage!!!
        return 0.0


def fetch_buildings(bbox):
    """
    Fetches and cleans building data.
    NOTE: If gpkg/bbox_dtla.gpkg is in the repo, it will only end up loading the bbox_dtla data.
    If you want something like the Manhattan scenario, delete bbox_dtla.gpkg and rerun.
    """
    local_path = "../gpkg/bbox_dtla.gpkg"
    
    if os.path.exists(local_path):
        print("Loading map data from local disk...")
        return gpd.read_file(local_path)
    else: 
        tags = {'building': True}
        gdf = ox.features_from_bbox(bbox=bbox, tags=tags) # type: ignore
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
        
        # Robust handling for 'building:levels' and 'height' to avoid Pylance errors
        levels_col = 'building:levels'
        height_col = 'height'
        
        if levels_col in gdf.columns:
            gdf['levels'] = pd.to_numeric(gdf[levels_col], errors='coerce')
        else:
            gdf['levels'] = np.nan
            
        if height_col in gdf.columns:
            gdf['height_meters'] = pd.to_numeric(gdf[height_col], errors='coerce')
        else:
            gdf['height_meters'] = np.nan
            
        # Fill defaults: Assume 3.5m per level if height is missing, or 2 levels as fallback
        gdf['levels'] = gdf['levels'].fillna(gdf['height_meters'] / 3.5).fillna(2)
        gdf['height_meters'] = gdf['levels'] * 3.5
        
        return gdf

def convert_to_local_origin(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    1. Projects the GeoDataFrame to a local UTM (meters) CRS.
    2. Shifts the geometry so the centroid of the entire collection is (0,0).
    """
    # 1. Project to UTM (meters)
    # estimate_utm_crs() automatically selects the correct UTM zone for your DTLA bbox
    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
    
    # 2. Calculate the centroid of the ENTIRE building collection
    # unary_union joins all polygons into one geometry object
    total_geometry = unary_union(gdf_utm.geometry)
    center = total_geometry.centroid
    
    # 3. Shift the geometry to (0,0)
    # We use .translate to move all coordinates relative to the calculated center
    gdf_utm['geometry'] = gdf_utm.geometry.translate(xoff=-center.x, yoff=-center.y)
    
    return gdf_utm

def calculate_d_2d_in(ue_pos, building_polygon: Polygon) -> float:
    """
    Calculates the penetration depth (d_2d_in) for O2I path loss.
    
    :param ue_pos: Vector object with x, y coordinates
    :param building_polygon: Shapely Polygon of the building
    :return: Shortest 2D distance to the nearest exterior wall (meters)
    """
    # Create a point for the UE
    ue_point = Point(ue_pos.x, ue_pos.y)
    
    # .exterior is the outer ring of the building polygon
    # .distance returns the shortest distance to the boundary
    d_2d_in = building_polygon.exterior.distance(ue_point)
    
    return d_2d_in

def get_building_meshes(gdf):
    """
    Converts a GeoDataFrame of buildings into Mesh3d-compatible dictionary,
    centered at the centroid of the entire building collection.
    """
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    
    # 1. Calculate the centroid of the ENTIRE dataset (not just the first building)
    total_geometry = unary_union(gdf.geometry)
    center = total_geometry.centroid
    
    vertex_offset = 0
    
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type != 'Polygon': continue
            
        height = row.get('height_meters', 3.0)
        if np.isnan(height): height = 3.0
            
        coords = np.array(geom.exterior.coords)[:-1]
        n = len(coords)
        
        # 2. Shift coordinates using the global centroid
        xs = coords[:, 0] - center.x
        ys = coords[:, 1] - center.y
        
        # Extrude: Base (z=0) and Top (z=height)
        all_x.extend(np.concatenate([xs, xs]))
        all_y.extend(np.concatenate([ys, ys]))
        all_z.extend(np.concatenate([np.zeros(n), np.full(n, height)]))
        
        # Wall Triangles
        for i in range(n):
            i1, i2, i3, i4 = vertex_offset+i, vertex_offset+(i+1)%n, vertex_offset+i+n, vertex_offset+(i+1)%n+n
            all_i.extend([i1, i2, i1]); all_j.extend([i2, i3, i3]); all_k.extend([i3, i4, i2])
            
        # Roof Triangles (Simple Fan)
        top_start = vertex_offset + n
        for i in range(1, n - 1):
            all_i.extend([top_start]); all_j.extend([top_start + i]); all_k.extend([top_start + i + 1])
            
        vertex_offset += 2 * n
        
    return {'x': all_x, 'y': all_y, 'z': all_z, 'i': all_i, 'j': all_j, 'k': all_k}

def is_link_blocked(ue_pos, gnb_pos, buildings_gdf):
    """
    Returns True if the line between UE and gNB passes through any building.
    """
    # Create a 3D line segment between UE and gNB
    link_line = LineString([(ue_pos.x, ue_pos.y, ue_pos.z), (gnb_pos.x, gnb_pos.y, gnb_pos.z)])
    
    # Check if this line intersects any of the building geometries
    # .intersects is very fast when using a GeoDataFrame's spatial index
    return buildings_gdf.intersects(link_line).any()

def generate_hex_grid(bounds, spacing):
    """
    Generates exactly 19 hexagonal grid points centered 
    within the provided bounds.
    """
    x_min, y_min, x_max, y_max = bounds
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    
    points = [center] # The center (1)
    
    # Ring 1: 6 points at distance 'spacing'
    for i in range(6):
        angle = i * (2 * np.pi / 6)
        points.append((center[0] + spacing * np.cos(angle),
                       center[1] + spacing * np.sin(angle)))
    
    # Ring 2: 12 points
    # 6 corners at distance 2 * spacing
    # 6 mid-points at distance sqrt(3) * spacing (30 degree offset)
    for i in range(6):
        # Corners
        angle = i * (2 * np.pi / 6)
        points.append((center[0] + 2 * spacing * np.cos(angle),
                       center[1] + 2 * spacing * np.sin(angle)))
        
        # Mid-points
        angle_mid = angle + (np.pi / 6)
        points.append((center[0] + np.sqrt(3) * spacing * np.cos(angle_mid),
                       center[1] + np.sqrt(3) * spacing * np.sin(angle_mid)))
        
    return np.array(points)

def nudge_to_outdoor(point, gdf, step=5, max_attempts=50, rng=np.random.default_rng):
    """
    Performs a spiral search around the point to find the first 
    outdoor location.
    """
    if not gdf.contains(point).any():
        return point.x, point.y
    
    # Spiral search: start close, push outward
    for i in range(1, max_attempts):
        # Sample points in a circle around the original grid point
        angle = rng.uniform(0, 2 * np.pi)
        distance = i * step # Expand radius by 'step' meters
        
        nx = point.x + distance * np.cos(angle)
        ny = point.y + distance * np.sin(angle)
        n_point = Point(nx, ny)
        
        # Check if this new spot is outdoor
        if not gdf.contains(n_point).any():
            return nx, ny
            
    return None # Failed to find a spot

