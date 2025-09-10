import geopandas as gpd
import json
import numpy as np
import leafmap.maplibregl as leafmap
import os
from shapely.geometry import LineString, MultiLineString

# ------------------------------------------------------------------
# BASE STYLE
# ------------------------------------------------------------------
style = {
    "version": 8,
    "sprite": "https://demotiles.maplibre.org/sprites/basic",  # Add this line to load the basic sprite
    "glyphs": "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    "sources": {
        "osm": {
            "type": "raster",
            "tiles": ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
            "tileSize": 256,
            "attribution": "© OpenStreetMap Contributors",
            "maxzoom": 19,
        },
        "terrainSource": {
            "type": "raster-dem",
            "url": "https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key=Cr28pZMZ7PIi8aDantUs",
            "tileSize": 256,
        },
        "hillshadeSource": {
            "type": "raster-dem",
            "url": "https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key=Cr28pZMZ7PIi8aDantUs",
            "tileSize": 256,
        },
    },
    "layers": [
        {"id": "osm", "type": "raster", "source": "osm"},
        {
            "id": "hills",
            "type": "hillshade",
            "source": "hillshadeSource",
            "layout": {"visibility": "visible"},
            "paint": {"hillshade-shadow-color": "#473B24"},
        },
    ],
    "terrain": {"source": "terrainSource", "exaggeration": 3},
}

# ------------------------------------------------------------------
# READ GPX OR CREATE SAMPLE
# ------------------------------------------------------------------

gdf = gpd.read_file("default_gpx.gpx", layer="tracks")

gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
gdf_4326 = gdf.to_crs(4326)
geojson_data = json.loads(gdf_4326.to_json())

style["sources"]["gpx-source"] = {"type": "geojson", "data": geojson_data}
style["layers"].append(
    {
        "id": "gpx-track",
        "type": "line",
        "source": "gpx-source",
        "layout": {"line-join": "round", "line-cap": "round"},
        "paint": {"line-color": "red", "line-width": 4, "line-opacity": 0.8},
    }
)

# ------------------------------------------------------------------
# MARKERS
# ------------------------------------------------------------------
def create_distance_markers(gdf, interval_km=2):
    # Extract the first geometry
    geom = gdf.iloc[0].geometry

    # Handle MultiLineString by taking the first part
    if isinstance(geom, MultiLineString):
        line = geom.geoms[0]  # Take the first LineString from the MultiLineString
    else:
        line = geom  # Assume it's a LineString

    # Ensure we have a LineString
    if not isinstance(line, LineString):
        raise ValueError("Geometry is not a LineString or MultiLineString.")

    total_length_km = line.length / 1000.0

    # Always include start and end points
    start_point = line.coords[0]  # First coordinate of the LineString
    end_point = line.coords[-1]   # Last coordinate of the LineString

    # Create GeoSeries for start and end points in the projected CRS
    start_point_proj = gpd.GeoSeries([gpd.points_from_xy([start_point[0]], [start_point[1]])[0]], crs=gdf.crs)
    end_point_proj = gpd.GeoSeries([gpd.points_from_xy([end_point[0]], [end_point[1]])[0]], crs=gdf.crs)

    # Convert to WGS84 for the final GeoJSON
    start_point_wgs84 = start_point_proj.to_crs(4326)[0]
    end_point_wgs84 = end_point_proj.to_crs(4326)[0]

    markers = []

    # Add START point (ID 0)
    markers.append(
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [start_point_wgs84.x, start_point_wgs84.y]},
            "properties": {"id": 0, "distance": "0.0 km", "elevation": "Start", "is_endpoint": True},
        }
    )

    # Add intermediate points at the specified interval
    next_id = 1
    for dist_km in np.arange(interval_km, total_length_km, interval_km):
        dist_m = dist_km * 1000.0
        point = line.interpolate(dist_m)
        point_wgs84 = gpd.GeoSeries([point], crs=gdf.crs).to_crs(4326)[0]
        elevation = 100 + (next_id * 10)  # placeholder
        markers.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [point_wgs84.x, point_wgs84.y]},
                "properties": {"id": next_id, "distance": f"{dist_km:.1f} km", "elevation": f"{elevation:.0f} m", "is_endpoint": False},
            }
        )
        next_id += 1

    # Add END point (ID = last intermediate ID + 1)
    markers.append(
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [end_point_wgs84.x, end_point_wgs84.y]},
            "properties": {"id": next_id, "distance": f"{total_length_km:.1f} km", "elevation": "End", "is_endpoint": True},
        }
    )

    return {"type": "FeatureCollection", "features": markers}

markers_geojson = create_distance_markers(gdf_proj)
style["sources"]["marker-source"] = {"type": "geojson", "data": markers_geojson}

# Add the marker circles (the blue dots)
style["layers"].append(
    {
        "id": "marker-circles",
        "type": "circle",
        "source": "marker-source",
        "paint": {
            "circle-radius": 10,
            "circle-color": "red",
            "circle-stroke-width": 2,
            "circle-stroke-color": "#ffffff",
        },
    }
)

# Add a SYMBOL layer for the CIRCLE ICON (background for the ID)
style["layers"].append(
    {
        "id": "marker-id-circle",
        "type": "symbol",
        "source": "marker-source",
        "layout": {
            "icon-image": "circle-15",  # Use the built-in circle icon
            "icon-size": 0.8,  # Adjust size as needed
            "icon-allow-overlap": True,
            "icon-ignore-placement": True,  # Allow it to overlap other symbols
        },
        "paint": {
            "icon-color": "#ffffff",  # White circle
            # "icon-halo-color": "#000000",  # Black border
            "icon-halo-width": 1,
        },
    }
)

# Add a SYMBOL layer for the ID TEXT (foreground, placed on top of the circle)
style["layers"].append(
    {
        "id": "marker-id-text",
        "type": "symbol",
        "source": "marker-source",
        "layout": {
            "text-field": ["get", "id"],  # Just the ID number
            "text-size": 12,
            "text-anchor": "center",  # Center the text on the point
            "text-allow-overlap": True,
            "text-ignore-placement": True,
        },
        "paint": {
            "text-color": "#ffffff",  # Black text for contrast
            "text-halo-color": "#ffffff",  # Optional: add a white halo for extra clarity
            "text-halo-width": 1,
        },
    }
)

# Add a SYMBOL layer for the DISTANCE and ELEVATION (below the ID circle)
style["layers"].append(
    {
        "id": "marker-labels",
        "type": "symbol",
        "source": "marker-source",
        "layout": {
            "text-field": [
                "format",
                ["get", "distance"], {"font-scale": 1.0},
                "\n", {},
                ["get", "elevation"], {"font-scale": 0.9, "text-color": "#666"}
            ],
            "text-size": 12,
            "text-offset": [0, 1.8],  # Position below the ID circle
            "text-anchor": "top",
            "text-allow-overlap": False,
        },
        "paint": {
            "text-color": "#000000",
            "text-halo-color": "#ffffff",
            "text-halo-width": 2,
            "text-halo-blur": 1,
        },
    }
)

# ------------------------------------------------------------------
# MAP INIT
# ------------------------------------------------------------------
centre_proj = gdf_proj.unary_union.centroid
centre_lonlat = gpd.GeoSeries([centre_proj], crs=gdf_proj.crs).to_crs(4326)[0].coords[0]

# Initialize the map
m = leafmap.Map(center=list(centre_lonlat), zoom=13, pitch=60, style=style)
m.add_layer_control(bg_layers=True)

# ------------------------------------------------------------------
# EXPORT HTML
# ------------------------------------------------------------------
html = m.to_html()

with open("3d_leafmap_terrain.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Map created: open 3d_leafmap_terrain.html")
