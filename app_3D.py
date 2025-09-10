import io
import json
import pytz
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from geopy import distance
import base64
import gpxpy
import geopandas as gpd
from shapely.geometry import LineString
import leafmap.maplibregl as leafmap

from gpx_converter import Converter
from sunrisesunset import SunriseSunset
from datetime import datetime, date, timedelta
from beaufort_scale.beaufort_scale import beaufort_scale_kmh
from timezonefinder import TimezoneFinder
tf = TimezoneFinder()

from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, callback, _dash_renderer
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import srtm
elevation_data = srtm.get_data()

import requests_cache
import openmeteo_requests
from retry_requests import retry

### VARIABLES ###

hdate_object = date.today()
hour = '10'
minute = '30'
speed = 4.0
frequency = 2

# Variables to become widgets
igpx = 'default_gpx.gpx'
hdate = hdate_object.strftime('%Y-%m-%d')
time = hour + ':' + minute
granularity = frequency * 1000

# Setup the Open Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Open Meteo weather forecast API
url = 'https://api.open-meteo.com/v1/forecast'
params = {
	'timezone': 'auto',
	'minutely_15': ['temperature_2m', 'rain', 'wind_speed_10m', 'weather_code', 'is_day'],
	'hourly': ['rain'],
}

# Load the JSON files mapping weather codes to descriptions and icons
with open('weather_icons_custom.json', 'r') as file:
	icons = json.load(file)

# Weather icons URL
icon_url = 'https://raw.githubusercontent.com/basmilius/weather-icons/refs/heads/dev/production/fill/svg/'
sunrise_icon = icon_url + 'sunrise.svg'
sunset_icon = icon_url + 'sunset.svg'

### FUNCTIONS ###

# Sunrise sunset
def sunrise_sunset(lat_start, lon_start, lat_end, lon_end, hdate):

    tz = tf.timezone_at(lng=lon_start, lat=lat_start)
    zone = pytz.timezone(tz)

    day = datetime.strptime(hdate, '%Y-%m-%d')

    dt = day.astimezone(zone)

    rs_start = SunriseSunset(dt, lat=lat_start, lon=lon_start, zenith='official')
    rise_time = rs_start.sun_rise_set[0]

    rs_end = SunriseSunset(dt, lat=lat_end, lon=lon_end, zenith='official')
    set_time = rs_end.sun_rise_set[1]

    sunrise = rise_time.strftime('%H:%M')
    sunset = set_time.strftime('%H:%M')

    return sunrise, sunset

# Map weather codes to descriptions and icons
def map_icons(df):
	code = df['weather_code']

	if df['is_day'] == 1:
		icon = icons[str(code)]['day']['icon']
		description = icons[str(code)]['day']['description']
	elif df['is_day'] == 0:
		icon = icons[str(code)]['night']['icon']
		description = icons[str(code)]['night']['description']

	df['Weather'] = icon_url + icon
	df['Weather outline'] = description

	return df

# Quantitative pluviometry to natural language
def rain_intensity(precipt):
    if precipt >= 50:
        rain = 'Extreme rain'
    elif 50  < precipt <= 16:
        rain = 'Very heavy rain'
    elif 4  <= precipt < 16:
        rain = 'Heavy rain'
    elif 1  <= precipt < 4:
        rain = 'Moderate rain'
    elif 0.25  <= precipt < 1:
        rain = 'Light rain'
    elif 0 < precipt < 0.25:
        rain = 'Light drizzle'
    else:
        rain = 'No rain / No info'
    return rain

# Function to add elevation
def add_ele(row):
    if pd.isnull(row['altitude']):
        row['altitude'] = elevation_data.get_elevation(row['latitude'], row['longitude'], 0)
    else:
        row['altitude'] = row['altitude']
    return row

# Compute distances using the Karney algorith with Euclidian altitude correction
def eukarney(lat1, lon1, alt1, lat2, lon2, alt2):
    p1 = (lat1, lon1)
    p2 = (lat2, lon2)
    karney = distance.distance(p1, p2).m
    return np.sqrt(karney**2 + (alt2 - alt1)**2)

# Obtain the weather forecast for each waypoint at each specific time
def get_weather(df_wp):

    params['latitude'] = df_wp['latitude']
    params['longitude'] = df_wp['longitude']
    params['elevation'] = df_wp['altitude']

    start_dt = datetime.strptime(hdate + 'T' + time, '%Y-%m-%dT%H:%M')

    delta_dt = start_dt + timedelta(seconds=df_wp['seconds'])
    delta_read = delta_dt.strftime('%Y-%m-%dT%H:%M')

    start_period = (delta_dt - timedelta(seconds=1800)).strftime('%Y-%m-%dT%H:%M')
    end_period = (delta_dt + timedelta(seconds=1800)).strftime('%Y-%m-%dT%H:%M')

    time_read = delta_dt.strftime('%H:%M')

    df_wp['Time'] = time_read

    params['start_minutely_15'] = delta_read
    params['end_minutely_15'] = delta_read
    params['start_hour'] = delta_read
    params['end_hour'] = delta_read

    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    minutely = response.Minutely15()
    hourly = response.Hourly()

    minutely_temperature_2m = minutely.Variables(0).ValuesAsNumpy()[0]
    rain = hourly.Variables(0).ValuesAsNumpy()[0]
    minutely_wind_speed_10m = minutely.Variables(2).ValuesAsNumpy()[0]
    weather_code = minutely.Variables(3).ValuesAsNumpy()[0]
    is_day = minutely.Variables(4).ValuesAsNumpy()[0]

    df_wp['Temp (°C)'] = minutely_temperature_2m
    df_wp['weather_code'] = weather_code
    df_wp['is_day'] = is_day

    v_rain_intensity = np.vectorize(rain_intensity)
    df_wp['Rain level'] = v_rain_intensity(rain)

    v_beaufort_scale_kmh = np.vectorize(beaufort_scale_kmh)

    df_wp['Wind level'] = v_beaufort_scale_kmh(minutely_wind_speed_10m, language='en')

    df_wp['Rain (mm/h)'] = rain.round(1)
    df_wp['Wind (km/h)'] = minutely_wind_speed_10m.round(1)

    return df_wp

# Parse the GPX track
def parse_gpx(df_gpx, hdate):

    # Sunrise sunset

    lat_start, lon_start = df_gpx[['latitude', 'longitude']].head(1).values.flatten().tolist()
    lat_end, lon_end = df_gpx[['latitude', 'longitude']].tail(1).values.flatten().tolist()

    sunrise, sunset = sunrise_sunset(lat_start, lon_start, lat_end, lon_end, hdate)

    df_gpx = df_gpx.apply(lambda x: add_ele(x), axis=1)

    centre_lat = (df_gpx['latitude'].max() + df_gpx['latitude'].min()) / 2
    centre_lon = (df_gpx['longitude'].max() + df_gpx['longitude'].min()) / 2

    # Create shifted columns in order to facilitate distance calculation

    df_gpx['lat_shift'] = df_gpx['latitude'].shift(periods=-1).fillna(df_gpx['latitude'])
    df_gpx['lon_shift'] = df_gpx['longitude'].shift(periods=-1).fillna(df_gpx['longitude'])
    df_gpx['alt_shift'] = df_gpx['altitude'].shift(periods=-1).fillna(df_gpx['altitude'])

    # Apply the distance function to the dataframe

    df_gpx['distances'] = df_gpx.apply(lambda x: eukarney(x['latitude'], x['longitude'], x['altitude'], x['lat_shift'], x['lon_shift'], x['alt_shift']), axis=1).fillna(0)
    df_gpx['distance'] = df_gpx['distances'].cumsum().round(decimals = 0).astype(int)

    df_gpx = df_gpx.drop(columns=['lat_shift', 'lon_shift', 'alt_shift', 'distances']).copy()

    start = df_gpx['distance'].min()
    finish = df_gpx['distance'].max()

    dist_rang = list(range(start, finish, granularity))
    dist_rang.append(finish)

    way_list = []
    for waypoint in dist_rang:
        gpx_dict = df_gpx.iloc[(df_gpx.distance - waypoint).abs().argsort()[:1]].to_dict('records')[0]
        way_list.append(gpx_dict)

    df_wp = pd.DataFrame(way_list)

    df_wp['seconds'] = df_wp['distance'].apply(lambda x: int(round(x / (speed * (5/18)), 0)))

    df_wp = df_wp.apply(lambda x: get_weather(x), axis=1)

    df_wp['Temp (°C)'] = df_wp['Temp (°C)'].round(0).astype(int).astype(str) + '°C'
    df_wp['is_day'] = df_wp['is_day'].astype(int)
    df_wp['weather_code'] = df_wp['weather_code'].astype(int)
    df_wp = df_wp.apply(map_icons, axis=1)

    df_wp['Rain level'] = df_wp['Rain level'].astype(str)
    df_wp['Wind level'] = df_wp['Wind level'].astype(str)

    df_wp = df_wp.reset_index(drop=True)

    df_wp['Waypoint'] = df_wp.index

    dfs = df_wp[['Waypoint', 'Time', 'Weather', 'Weather outline', 'Temp (°C)', 'Rain (mm/h)', 'Rain level', 'Wind (km/h)', 'Wind level']].copy()

    dfs['Wind (km/h)'] = dfs['Wind (km/h)'].round(1).astype(str).replace('0.0', '')
    dfs['Rain (mm/h)'] = dfs['Rain (mm/h)'].round(1).astype(str).replace('0.0', '')
    dfs['Temp (°C)'] = dfs['Temp (°C)'].str.replace('C', '')

    dfs['Weather'] = '<img style="float: right; padding: 0; margin: -6px; display: block;" width=48px; src=' + dfs['Weather'] + '>'

    return df_gpx, df_wp, dfs, sunrise, sunset, centre_lat, centre_lon

### PLOTS ###

# Plot 3D relief map
def plot_3d_map(df_gpx, df_wp, centre_lat, centre_lon):
    # BASE STYLE
    style = {
        "version": 8,
        "sprite": "https://demotiles.maplibre.org/sprites/basic",  # Add this line
        "glyphs": "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
        "sources": {
            "osm": {
                "type": "raster",
                "tiles": ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
                "tileSize": 256,
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

    # Convert GPX DataFrame to GeoJSON
    line = LineString(zip(df_gpx['longitude'], df_gpx['latitude']))
    gdf_4326 = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[line])
    geojson_data = json.loads(gdf_4326.to_json())
    style["sources"]["gpx-source"] = {"type": "geojson", "data": geojson_data}
    style["layers"].append({
        "id": "gpx-track",
        "type": "line",
        "source": "gpx-source",
        "layout": {"line-join": "round", "line-cap": "round"},
        "paint": {"line-color": "red", "line-width": 4, "line-opacity": 0.8},
    })

    # Create markers GeoJSON from waypoint DataFrame
    markers = []
    for index, row in df_wp.iterrows():
        markers.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row['longitude'], row['latitude']]},
            "properties": {
                "id": row['Waypoint'],
                "weather_outline": row['Weather outline'],
                "distance_km": f"{(row['distance'] / 1000.0):.1f} km",
                "altitude_m": f"{int(row['altitude'])} m"
            },
        })
    markers_geojson = {"type": "FeatureCollection", "features": markers}
    style["sources"]["marker-source"] = {"type": "geojson", "data": markers_geojson}

    # Add marker layers - in correct order
    style["layers"].extend([
        {
            "id": "marker-circles",
            "type": "circle",
            "source": "marker-source",
            "paint": {
                "circle-radius": 10,
                "circle-color": "firebrick",
                "circle-stroke-width": 2,
                "circle-stroke-color": "#ffffff",
            },
        },
        {
            "id": "marker-id-circle",
            "type": "symbol",
            "source": "marker-source",
            "layout": {
                "icon-image": "circle-15",
                "icon-size": 0.8,
                "icon-allow-overlap": True,
                "icon-ignore-placement": True,
            },
            "paint": {
                "icon-color": "#ffffff",
                "icon-halo-width": 1,
            },
        },
        {
            "id": "marker-id-text",
            "type": "symbol",
            "source": "marker-source",
            "layout": {
                "text-field": ["get", "id"],
                "text-size": 12,
                "text-anchor": "center",
                "text-allow-overlap": True,
                "text-ignore-placement": True,
            },
            "paint": {"text-color": "#000000"},
        },
        {
            "id": "marker-labels",
            "type": "symbol",
            "source": "marker-source",
            "layout": {
                "text-field": [
                    "format",
                    ["get", "weather_outline"], {"font-scale": 1.0},
                    "\n", {},
                    ["get", "distance_km"], {"font-scale": 1.0},
                    "\n", {},
                    ["get", "altitude_m"], {"font-scale": 1.0},
                ],
                "text-size": 12,
                "text-offset": [0, 2.5],
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
    ])

    # Initialize map and add layer control
    m = leafmap.Map(center=[centre_lon, centre_lat], zoom=12, pitch=60, style=style)
    m.add_layer_control(bg_layers=True)

    # Save HTML for debugging
    html_content = m.to_html(read_only=True)
    with open("debug_map.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content

### DASH APP ###

external_stylesheets = [dbc.themes.BOOTSTRAP, dmc.styles.ALL]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Layout
hours = [str(n).zfill(2) for n in range(0, 24)]
minutes = [str(n).zfill(2) for n in range(0, 60, 5)]

picker_style = {
    'display': 'inline-block',
    'width': '35px',
    'height': '32px',
    'cursor': 'pointer',
    'border': 'none',
}

def serve_layout():
    layout = html.Div([
        html.Div([dcc.Link('The Weather for Hikers', href='.',
                style={'color': 'darkslategray', 'font-size': 18, 'font-family': 'sans', 'font-weight': 'bold', 'text-decoration': 'none'}),
        ]),
        html.Div([dcc.Link('Freedom Luxembourg', href='https://www.freeletz.lu/freeletz/',
                target='_blank', style={'color': 'goldenrod', 'font-size': 14, 'font-family': 'sans', 'text-decoration': 'none'}),
        ]),
        html.Div([html.Br(),
        dbc.Row([
            dbc.Col([dcc.Upload(id='upload-gpx', children=html.Div(id='name-gpx'),
                                accept='.gpx, .GPX', max_size=10000000, min_size=100,
                                            style={
                                                'width': '174px',
                                                'height': '48px',
                                                'lineWidth': '174px',
                                                'lineHeight': '48px',
                                                'borderWidth': '2px',
                                                'borderStyle': 'solid',
                                                'borderColor': 'goldenrod',
                                                'textAlign': 'center',
                                                },
                                    ), dcc.Store(id='store-gpx')], width={'size': 'auto', 'offset': 1}),
            dbc.Col([dmc.Divider(orientation='vertical', size=2, color='goldenrod', style={'height': 82})], width={'size': 'auto'}),
            dbc.Col([dbc.Label('Date of the hike'), html.Br(),
                dcc.DatePickerSingle(id='calendar-date',
                placeholder='Select the date of your hike',
                display_format='Do MMMM YYYY',
                min_date_allowed=date.today(),
                max_date_allowed=date.today() + timedelta(days=7),
                initial_visible_month=date.today(),
                date=date.today()), dcc.Store(id='store-date')], width={'size': 'auto'}),
            dbc.Col([dmc.Divider(orientation='vertical', size=2, color='goldenrod', style={'height': 82})], width={'size': 'auto'}),
            dbc.Col([html.Div([html.Label('Start time'), html.Br(), html.Br(),
                            html.Div([dcc.Dropdown(hours, placeholder=hour, value=hour, style=picker_style, id='dropdown-hour'),
                                    dcc.Store(id='store-hour'),
                                    html.Span(':'),
                                    dcc.Dropdown(minutes, placeholder=minute, value=minute, style=picker_style, id='dropdown-minute'),
                                    dcc.Store(id='store-minute')],
                                style={'border': '1px solid goldenrod',
                                    'height': '34px',
                                    'width': '76px',
                                    'display': 'flex',
                                    'align-items': 'center',
                                },
                            ),
                        ], style={'font-family': 'Sans'},
                    ),
                ], width={'size': 'auto'}),
            dbc.Col([dmc.Divider(orientation='vertical', size=2, color='goldenrod', style={'height': 82})], width={'size': 'auto'}),
            dbc.Col([dbc.Label('Average pace (km/h)'), html.Div(dcc.Slider(3, 6.5, 0.5, value=speed, id='slider-pace'), style={'width': '272px'}), dcc.Store(id='store-pace')], width={'size': 'auto'}),
            dbc.Col([dmc.Divider(orientation='vertical', size=2, color='goldenrod', style={'height': 82})], width={'size': 'auto'}),
            dbc.Col([dbc.Label('Forecast frequency (km)'), html.Div(dcc.Slider(1, 5, 1, value=frequency, id='slider-freq'), style={'width': '170px'}), dcc.Store(id='store-freq')], width={'size': 'auto'}),
            dbc.Col([dmc.Divider(orientation='vertical', size=2, color='goldenrod', style={'height': 82})], width={'size': 'auto'}),
            dbc.Col([html.Br(), html.Button('Forecast', id='submit-forecast', n_clicks=0,
                                            style={'width': '86px', 'height': '36px', 'background-color': 'goldenrod', 'font-weight': 'bold', 'color': 'white'})],
            width={'size': 'auto'}),
            ]),
        ], style={'font-size': 13, 'font-family': 'sans'}),
        html.Div([html.Br(),
        dbc.Row([dbc.Col(html.Div('Sunrise '), width={'size': 'auto', 'offset': 9}),
                dbc.Col(html.Img(src=sunrise_icon, style={'height':'42px'}), width={'size': 'auto'}),
                dbc.Col(html.Div(id='sunrise-time'), width={'size': 'auto'}),
                dbc.Col([dmc.Divider(orientation='vertical', size=2, color='goldenrod', style={'height': 22})], width={'size': 'auto'}),
                dbc.Col(html.Div('Sunset '), width={'size': 'auto', 'offset': 0}),
                dbc.Col(html.Img(src=sunset_icon, style={'height':'42px'}), width={'size': 'auto'}),
                dbc.Col(html.Div(id='sunset-time'), width={'size': 'auto'})]),
        ], style={'font-size': 13, 'font-family': 'sans'}),
        html.Div(id='datatable-div'),
        html.Div(id='base-figure-div', style={'height': '90vh'}),
        html.Div([dcc.Link('Freedom Luxembourg', href='https://www.freeletz.lu/freeletz/',
                target='_blank', style={'color': 'goldenrod', 'font-size': 15, 'font-family': 'sans', 'text-decoration': 'none'}),
        ], style={'text-align': 'center'},),
        html.Div([dcc.Link('Powered by Open Meteo', href='https://open-meteo.com/',
                target='_blank', style={'color': 'darkslategray', 'font-size': 13, 'font-family': 'sans', 'text-decoration': 'none'}),
        ], style={'text-align': 'center'}),
        dcc.Interval(
                    id='interval-component',
                    interval=6 * 60 * 60 * 1000,
                    n_intervals=0),
    ], id='layout-content')

    layout = dmc.MantineProvider(layout)
    return layout

app.layout = serve_layout

# Callbacks
@callback(Output('store-gpx', 'data'),
          Output('name-gpx', 'children'),
    Input('upload-gpx', 'contents'),
    State('upload-gpx', 'filename'))
def update_gpx(contents, filename):
    if filename:
        try:
            igpx = filename
            message = html.Div(['Upload your GPX track ', html.H6(igpx, style={'color': 'darkslategray', 'font-size': 12, 'font-weight': 'bold'})])
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            gpx_parsed = gpxpy.parse(decoded)
            points = []
            for track in gpx_parsed.tracks:
                for segment in track.segments:
                    for p in segment.points:
                        points.append({
                            'latitude': p.latitude,
                            'longitude': p.longitude,
                            'altitude': p.elevation,
                    })
            df_gpx = pd.DataFrame.from_records(points)
        except Exception:
            igpx = 'default_gpx.gpx'
            message = html.Div(['Upload your GPX track ', html.H6('The GPX cannot be parsed. Please, upload another file.', style={'color': 'darkslategray', 'font-size': 12, 'font-weight': 'bold'})])
            df_gpx = Converter(input_file = igpx).gpx_to_dataframe()
    else:
        igpx = 'default_gpx.gpx'
        message = html.Div(['Upload your GPX track ', html.H6(igpx, style={'color': 'darkslategray', 'font-size': 12, 'font-weight': 'bold'})])
        df_gpx = Converter(input_file = igpx).gpx_to_dataframe()

    return df_gpx.to_dict('records'), message

@callback(Output('store-date', 'data'),
    Input('calendar-date', 'date'))
def update_date(value):
    return value or hdate

@callback(Output('store-hour', 'data'),
    Input('dropdown-hour', 'value'))
def update_hour(value):
    return value or hour

@callback(Output('store-minute', 'data'),
    Input('dropdown-minute', 'value'))
def update_minute(value):
    return value or minute

@callback(Output('store-freq', 'data'),
    Input('slider-freq', 'value'))
def update_freq(value):
    return value or frequency

@callback(Output('store-pace', 'data'),
    Input('slider-pace', 'value'))
def update_pace(value):
    return value or speed

@callback(Output('sunrise-time', 'children'),
          Output('sunset-time', 'children'),
          Output('datatable-div', 'children'),
          Output('base-figure-div', 'children'),
    Input('submit-forecast', 'n_clicks'),
    State('store-gpx', 'data'),
    State('store-date', 'data'),
    State('store-hour', 'data'),
    State('store-minute', 'data'),
    State('store-freq', 'data'),
    State('store-pace', 'data'),
    prevent_initial_call=False)
def weather_forecast(n_clicks, gpx_json, cdate, h, m, freq, pace):
    global df_wp, hdate, hour, minute, time, frequency, granularity, speed

    hdate = cdate or hdate
    hour = h or hour
    minute = m or minute
    time = f"{hour}:{minute}"
    frequency = freq or frequency
    granularity = frequency * 1000
    speed = pace or speed

    if not gpx_json:
        df_gpx = Converter(input_file='default_gpx.gpx').gpx_to_dataframe()
        gpx_json = df_gpx.to_dict('records')

    if n_clicks >= 0:
        gpx_df = pd.DataFrame.from_records(gpx_json)
        df_gpx, df_wp, dfs, sunrise, sunset, centre_lat, centre_lon = parse_gpx(gpx_df, hdate)

        sunrise_div = html.Div([sunrise])
        sunset_div = html.Div([sunset])

        table_div = html.Div([dash_table.DataTable(id='datatable-display',
            markdown_options={'html': True},
            columns=[{'name': i, 'id': i, 'deletable': False, 'selectable': False, 'presentation': 'markdown'} for i in dfs.columns],
            data=dfs.to_dict('records'),
            editable=False,
            row_deletable=False,
            style_as_list_view=True,
            style_cell={'fontSize': '12px', 'text-align': 'center', 'margin-bottom':'0'},
            css=[dict(selector='p', rule='margin: 0; text-align: center')],
            style_header={'backgroundColor': 'goldenrod', 'color': 'white', 'fontWeight': 'bold'})
        ])

        map_html = plot_3d_map(df_gpx, df_wp, centre_lat, centre_lon)
        figure_div = html.Iframe(srcDoc=map_html, style={'height': '100%', 'width': '100%', 'border': 'none'})

        return sunrise_div, sunset_div, table_div, figure_div

    return no_update, no_update, no_update, no_update


@callback(Output('layout-content', 'children'),
        [Input('interval-component', 'n_intervals')])
def refresh_layout(n):
    return serve_layout()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
