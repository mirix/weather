import os
import json
import pathlib
import gpxpy
import pandas as pd
import gradio as gr
from datetime import datetime

import pytz
from sunrisesunset import SunriseSunset
from timezonefinder import TimezoneFinder
tf = TimezoneFinder()
from beaufort_scale.beaufort_scale import beaufort_scale_ms

from wordsegment import load, segment
load()

import srtm
elevation_data = srtm.get_data()

import requests

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent='FreeLetzWeather')

from apscheduler.schedulers.background import BackgroundScheduler

### Default variables ###

# Met No weather forecast API
url = 'https://api.met.no/weatherapi/locationforecast/2.0/complete'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

# Weather icons URL
icon_url = 'https://raw.githubusercontent.com/metno/weathericons/89e3173756248b4696b9b10677b66c4ef435db53/weather/svg/'

# Custom CSS
css = '''
#button {background: DarkGoldenrod;}
.buttons {color: white;}
#table {height: 1080px;}
.tables {height: 1080px;}

.required-dropdown input:focus {
    color: white;
    background-color: DarkGoldenrod;
    box-shadow: 0 0 0 12px DarkGoldenrod;
}
'''

# Default GPX if none is uploaded
gpx_file = os.path.join(os.getcwd(), 'default_gpx.gpx')
gpx_path = pathlib.Path(gpx_file)


### Functions ###

# Pluviometry to natural language
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
        rain = ''
    return rain

def gen_dates_list():

    global day_print
    global dates_filt
    global dates_dict
    global dates_list
    global day_read
    global today

    today = datetime.today()
    day_read = today.strftime('%A %-d %B')
    day_print = '<h2>' + day_read + '</h2>'

    resp = requests.get(url=url, headers=headers, params=params)
    data = resp.json()

    dates_aval = []
    for d in data['properties']['timeseries']:
        if 'next_1_hours' in d['data']:
            date = datetime.strptime(d['time'], '%Y-%m-%dT%H:%M:%SZ')
            dates_aval.append(date.date())

    dates_aval = sorted(set(dates_aval))

    if len(dates_aval) > 3:
        dates_aval = dates_aval[:3]

    dates_read = [x.strftime('%A %-d %B %Y') for x in dates_aval]
    dates_filt = [x.strftime('%Y-%m-%d') for x in dates_aval]


    dates_dict = dict(zip(dates_read, dates_filt))
    dates_list = list(dates_dict.keys())

    return dates_list

def sunrise_sunset(lat, lon, day):

    tz = tf.timezone_at(lng=lon, lat=lat)
    zone = pytz.timezone(tz)

    dt = day.astimezone(zone)

    rs = SunriseSunset(dt, lat=lat, lon=lon, zenith='official')
    rise_time, set_time = rs.sun_rise_set

    sunrise = rise_time.strftime('%H:%M')
    sunset = set_time.strftime('%H:%M')

    sunrise_icon = '<img style="float: left;" width="24px" src=' + icon_url + 'clearsky_day.svg>'
    sunset_icon = '<img style="float: left;" width="24px" src=' + icon_url + 'clearsky_night.svg>'

    sunrise = '<h6>' + sunrise_icon  + ' Sunrise ' + sunrise + '</h6>'
    sunset = '<h6>' + sunset_icon + ' Sunset ' + sunset + '</h6>'

    return sunrise, sunset

# Download the JSON and filter it per date
def json_parser(date):

    global dfs

    resp = requests.get(url=url, headers=headers, params=params)
    data = resp.json()

    day = date

    dict_weather = {'Time': [], 'Weather': [], 'Weather outline': [], 'Temp (°C)': [], 'Rain (mm/h)': [], 'Rain level': [], 'Wind (m/s)': [], 'Wind level': [] }

    for d in data['properties']['timeseries']:
        date = datetime.strptime(d['time'], '%Y-%m-%dT%H:%M:%SZ')
        date_read = date.strftime('%Y-%m-%d')

        if date_read == day:
            dict_weather['Time'].append(date.strftime('%H'))
            weather = d['data']['next_1_hours']['summary']['symbol_code']
            icon_path = '<img style="float: left; padding: 0; margin: -6px; display: block;" width=32px; src=' + icon_url + weather + '.svg>'
            dict_weather['Weather'].append(icon_path)
            weather_read = ' '.join(segment(weather.replace('_', '')))
            dict_weather['Weather outline'].append(weather_read)
            temp = d['data']['instant']['details']['air_temperature']
            dict_weather['Temp (°C)'].append(str(int(round(temp, 0))) + '°')
            rain = d['data']['next_1_hours']['details']['precipitation_amount']
            dict_weather['Rain (mm/h)'].append(rain)
            dict_weather['Rain level'].append(rain_intensity(rain))
            wind = d['data']['instant']['details']['wind_speed']
            dict_weather['Wind (m/s)'].append(wind)
            dict_weather['Wind level'].append(beaufort_scale_ms(wind, language='en'))

    df = pd.DataFrame(dict_weather)

    df['Weather outline'] = df['Weather outline'].str.capitalize().str.replace(' night','').str.replace(' day','')
    df[['Rain (mm/h)', 'Wind (m/s)']] = df[['Rain (mm/h)', 'Wind (m/s)']].round(1).replace({0:''}).astype(str)

    dfs = df.style.set_properties(**{'border': '0px'})

    return dfs

# Extract coordinates and location from GPX file
def coor_gpx(gpx):

    def parse_gpx(gpx):

        global gpx_name
        global params
        global lat
        global lon
        global altitude
        global location
        global dates_dict
        global dates_list
        global day_read
        global dates
        global sunrise
        global sunset

        with open(gpx.name) as f:
            gpx_parsed = gpxpy.parse(f)
        # Convert to a dataframe one point at a time.
        points = []
        for track in gpx_parsed.tracks:
            for segment in track.segments:
                for p in segment.points:
                    points.append({
                        'lat': p.latitude,
                        'lon': p.longitude,
                        'altitude': p.elevation,
                })
        df_gpx = pd.DataFrame.from_records(points)
        params = df_gpx.iloc[-1].to_dict()
        lat = params['lat']
        lon = params['lon']

        if params['altitude'] == None:
            params['altitude'] = int(round(elevation_data.get_elevation(lat, lon), 0))
        else:
            params['altitude'] = int(round(params['altitude'], 0))

        altitude = params['altitude']

        location = geolocator.reverse('{}, {}'.format(lat, lon), zoom=14)

        gpx_name = 'You have uploaded <b style="color: #004170;">' + os.path.basename(gpx.name) + '</b>'
        location = '<p style="color: #004170">' + str(location) + '</p>'

        dates_list = gen_dates_list()
        day_read = dates_list[0]
        date_filt = datetime.strptime(day_read, '%A %d %B %Y')
        date_filt = date_filt.strftime('%Y-%m-%d')
        day_print = '<h2>' + day_read + '</h2>'

        sunrise, sunset = sunrise_sunset(lat, lon, datetime.strptime(day_read, '%A %d %B %Y'))

        dates = gr.Dropdown(choices=dates_list, label='2. Next, pick up the date of your hike', value=dates_list[0], interactive=True, elem_classes='required-dropdown')

        dfs = json_parser(date_filt)

    try:
        parse_gpx(gpx)
    except:
        parse_gpx(gpx_path)
        global gpx_name
        gpx_name = '<b style="color: firebrick;">ERROR: Not a valid GPX file. Upload another file.</b>'

    return gpx_name, location, dates, day_print, sunrise, sunset, dfs

# Choose a date from the dropdown menu
def date_chooser(day):
    global day_read
    global sunrise
    global sunset
    global sunrise_icon
    global sunset_icon
    global dates_dict
    global dates_list

    dates_list = gen_dates_list()

    day_read = day
    day_print = '<h2>' + day_read + '</h2>'

    date = datetime.strptime(day, '%A %d %B %Y')

    index = dates_list.index(day)

    sunrise, sunset = sunrise_sunset(lat, lon, date)

    date_filt = date.strftime('%Y-%m-%d')
    dfs = json_parser(date_filt)

    dates = gr.Dropdown(choices=dates_list, label='2. Next, pick up the date of your hike', value=dates_list[index], interactive=True, elem_classes='required-dropdown')

    return day_print, sunrise, sunset, dfs, dates

# Call functions with default values
coor_gpx(gpx_path)
sunrise, sunset = sunrise_sunset(lat, lon, today)
dfs = json_parser(dates_filt[0])

### Gradio app ###
with gr.Blocks(theme='ParityError/Interstellar', css=css, fill_height=True) as demo:
    with gr.Column():
        with gr.Row():
            gr.HTML('<h1 style="color: DarkGoldenrod">Freedom Luxembourg<br><h3 style="color: #004170">The Weather for Hikers</h3></h1>')
            with gr.Column():
                upload_gpx = gr.UploadButton(label='1. Upload your GPX track', file_count='single', size='lg', file_types=['.gpx', '.GPX'], elem_id='button', elem_classes='buttons', interactive=True)
                file_name = gr.HTML('<h6>' + gpx_name + '</h6>')
            dates = gr.Dropdown(choices=gen_dates_list(), label='2. Pick up the date of your hike', value=dates_list[0], interactive=True, elem_classes='required-dropdown')
        gr.HTML('<h1><br></h1>')
        with gr.Row():
            choosen_date = gr.HTML(day_print)
            loc = gr.HTML('<p style="color: #004170">' + str(location) + '</p>')
            sunrise = gr.HTML(sunrise)
            sunset = gr.HTML(sunset)
    table = gr.DataFrame(dfs, max_height=1000, type='pandas', headers=None, line_breaks=False, interactive=False, wrap=True, visible=True, render=True,
            elem_id='table', elem_classes='tables',
            datatype=['str', 'html', 'str', 'str', 'str', 'str', 'str', 'str'],
            )
    gr.HTML('<center>Freedom Luxembourg<br><a style="color: DarkGoldenrod; font-style: italic; text-decoration: none" href="https://www.freeletz.lu/freeletz/" target="_blank">freeletz.lu</a></center>')
    gr.HTML('<center>Powered by the <a style="color: #004170; text-decoration: none" href="https://api.met.no/weatherapi/locationforecast/2.0/documentation" target="_blank">Norwegian Meteorological Institute</a> API</center>')
    upload_gpx.upload(fn=coor_gpx, inputs=upload_gpx, outputs=[file_name, loc, dates, choosen_date, sunrise, sunset, table])
    dates.input(fn=date_chooser, inputs=dates, outputs=[choosen_date, sunrise, sunset, table, dates])

def restart_app():
    demo.close()
    port = int(os.environ.get('PORT', 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

scheduler = BackgroundScheduler({'apscheduler.timezone': 'Europe/Luxembourg'})
scheduler.add_job(func=restart_app, trigger='cron', hour='05', minute='55')
scheduler.start()

port = int(os.environ.get('PORT', 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
