import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from PIL import Image
import scipy
from skimage.io import imread, imshow

def masker(image, masks, count):
    fig, axes = plt.subplots(1, count, figsize=(12, 10))
    image_copy = image.copy()
    for n, ax in enumerate(axes.flatten()):
        masked_image = np.dstack((image_copy[:, :, 0]*(masks==[n]),
                                  image_copy[:, :, 1]*(masks==[n]),
                                  image_copy[:, :, 2]*(masks==[n])))
        ax.imshow(masked_image)
        ax.set_title(f'Cluster : {n+1}', fontsize = 20)
        ax.set_axis_off()
    fig.tight_layout() 
    plt.show()  
    

if __name__ == '__main__':

    image = imread("test/img_20200611_102834_043.jpg")
    h,w,c = image.shape
    data = image.reshape(h*w, c)
    gmm = mixture.GaussianMixture(n_components=4, covariance_type="tied")
    gmm = gmm.fit(data)

    cluster = gmm.predict(data)
    cluster = cluster.reshape(h, w)

    masker(image,cluster, gmm.n_components)

