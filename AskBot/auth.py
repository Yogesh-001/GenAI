import os
import urllib.request
from base64 import b64encode

username = "Yogesh.Murala"
password = os.getenv("DOCUPEDIA_ACCESS_TOKEN")

auth = b64encode(f"{username}:{password}".encode()).decode("utf-8")

url = "https://insidedocupedia.bosch.com/confluence/display/AOS/Nvidia+DMC+POC+conclusion+Points"
headers = {"Authorization": f"Basic {auth}"}

request = urllib.request.Request(url, headers=headers)
try:
    response = urllib.request.urlopen(request)
    result = response.read()
    print(result.decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
except Exception as e:
    print(f"Error: {str(e)}")
