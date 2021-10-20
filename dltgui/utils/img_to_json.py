import numpy as np
import json

# imh -> PIL Image
def img2json(img, json_path):
	s = img.tobytes().decode('latin1')
	with open(json_path, 'w+') as file:
		json.dump(s, file)