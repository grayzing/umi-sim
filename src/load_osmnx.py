from utils import fetch_buildings, bbox_dtla

gdf = fetch_buildings(bbox_dtla)
gdf.to_file("dtla_gdf.gpkg", driver="GPKG")