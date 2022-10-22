#%%
from matplotlib import image
from itertools import product

original = image.imread("original.png")
image_size = 64 

r_c =  list(product(range(original.shape[0]//image_size), range(original.shape[1]//image_size)))
for i, (r, c) in enumerate(r_c):
    one_image = original[
        image_size * r : image_size * (r+1), 
        image_size * c : image_size * (c+1)]
    image.imsave("{}.png".format(i), one_image)
# %%
