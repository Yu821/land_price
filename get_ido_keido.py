import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

URL = 'http://www.geocoding.jp/api/'


def coordinate(address):
	"""
	addressに住所を指定すると緯度経度を返す。

	>>> coordinate('東京都文京区本郷7-3-1')
	['35.712056', '139.762775']
	"""
	payload = {'q': address}
	html = requests.get(URL, params=payload)
	soup = BeautifulSoup(html.content, "html.parser")
	if soup.find('error'):
		# raise ValueError(f"Invalid address submitted. {address}")
		latitude = 'nan'
		longitude = 'nan'
		print('ERROR !!!')
		return [latitude, longitude]

	latitude = soup.find('lat').string
	longitude = soup.find('lng').string
	return [latitude, longitude]


def coordinates(addresses, interval=11, progress=True):
	"""
	addressesに住所リストを指定すると、緯度経度リストを返す。

	>>> coordinates(['東京都文京区本郷7-3-1', '東京都文京区湯島３丁目３０−１'], progress=False)
	[['35.712056', '139.762775'], ['35.707771', '139.768205']]
	"""
	coordinates = []
	for address in progress and tqdm(addresses) or addresses:
		coordinates.append(coordinate(address))
		time.sleep(interval)
	return coordinates


def get_not_found_jukyo(jukyo):

	jukyo = jukyo[jukyo['keido']=='見つかりません']

	jukyo.reset_index(drop=True, inplace=True)

	return jukyo


jukyo = pd.read_csv('./dataset/ido_keido.csv')

# Get not_found_jukyo
jukyo = get_not_found_jukyo(jukyo)

# Make new jukyo with not found
new_jukyo = pd.DataFrame(data={'jukyo':jukyo['jukyo']},index=range(jukyo.shape[0]))

# Get coordinates
coordinates = coordinates(jukyo['jukyo'].values)

coordinates = pd.DataFrame(coordinates,columns=['ido','keido'])
coordinates = pd.concat([coordinates,new_jukyo],axis=1)

# Save coordinates and jukyo not found in csv file
coordinates.to_csv('./dataset/coordinates_and_jukyo.csv',index=False)

