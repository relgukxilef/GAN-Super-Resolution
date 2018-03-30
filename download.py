from derpibooru import Search, sort
import requests
import scipy.misc as scm
from glob import glob

image_path = "data/images/"

def download(id, url):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(image_path + "{}.png".format(id), 'wb') as f:
            for chunk in r:
                f.write(chunk)

query = "width.gt:1024, height.gt:1024, -animated"
print(query)

existing_files = [f[len(image_path):] for f in glob(image_path + "*")]

for image in Search().query(query).sort_by(sort.SCORE).limit(100000):
    try:
        if "{}.png".format(image.id) not in existing_files:
            download(image.id, image.full)
        else:
            print('.', end='', flush=True)
    except (IndexError, AttributeError):
        print("failed at image " + image.id)
