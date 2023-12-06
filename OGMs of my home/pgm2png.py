import os
from PIL import Image



for file in os.listdir():
    filename, extension  = os.path.splitext(file)
    if extension == "1.pgm":
        new_file = "{}.png"
        with Image.open(file) as im:
            im.save(new_file)
