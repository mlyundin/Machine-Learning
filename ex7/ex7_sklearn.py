import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.cluster import KMeans

if __name__ == '__main__':

    img=mpimg.imread('bird_small.png')
    imgplot = plt.imshow(img)
    plt.show()

    img_size = img.shape
    X = img.reshape(img_size[0]*img_size[1], img_size[2])
    K = 16
    max_iters = 10

    km = KMeans(K, max_iter=max_iters)
    km.fit(X)

    X_recovered = km.cluster_centers_[km.labels_, :]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img)
    ax2.imshow(X_recovered.reshape(img_size))

    plt.show()