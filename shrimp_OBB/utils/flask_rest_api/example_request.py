"""Perform test request"""
import pprint

import requests

DETECTION_URL = "http://localhost:5003/"
TEST_IMAGE = r"/home/lee/Work/data/shrimp_video_new/2022-03-03-23_04_32.mp4"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
