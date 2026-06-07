import math
import random

import cv2
from tetra3 import tetra3
import numpy as np
from PIL import Image
#在此处新增import
from make_star.make_star_pure import create_star_image_pure
from make_star.make_star_move import create_star_image_move
from make_star.make_star_pollute import create_star_image_pollute

# ra = 12
# dec = 87
# rotation = 18
# fov = 21.32
size = (1960,1080)
speed = 1

#分别用三个函数生成三张图片
# pure_image = create_star_image_pure(ra, dec, rotation, fov, size[0], size[1],show_star=1)
# move_image = create_star_image_move(ra, dec, rotation, fov, size[0], size[1], speed=speed,show_star=3)
# pollute_image = create_star_image_pollute(ra, dec, rotation, fov, size[0], size[1],show_star=1)
#
# cv2.imwrite(r'E:/Code/my_star/test_image/test_pure.jpg', pure_image)
# cv2.imwrite(r'E:/Code/my_star/test_image/test_move.jpg', move_image)
# cv2.imwrite(r'E:/Code/my_star/test_image/test_pollute.jpg', pollute_image)


t3 = tetra3.Tetra3(load_database='gaia5-40')
# image = Image.open(r'E:/Code/my_star/dataset/raw_images/001.png')
# result = t3.solve_from_image(image)
# print(result)
ans = 0
for i in range(100):
    ra = random.uniform(0,360)
    dec = random.uniform(-90,90)
    rotation = random.uniform(0,360)
    fov = random.uniform(5,40)
    test_image = create_star_image_pure(ra, dec, rotation, fov, size[0], size[1],show_star=1)
    rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    result = t3.solve_from_image(pil_image)
    if (result['RA'] == None):
        print(i, ra, dec, rotation, fov)
        ans = ans + 1
        # break
    if (i%100 == 0):
        print(i)
print(ans)