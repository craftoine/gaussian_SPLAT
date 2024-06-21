#output.txt is a fine containing set of points in 3D
#this script will visualize the points in 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt('output.txt')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[::10,0], data[::10,1], data[::10,2], marker='o')
plt.show()

#output2.txt is a file containing set of balls in 3D given with a opacity value
#this script will visualize the balls in 3D
data = np.loadtxt('output2.txt')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0,data.shape[0],30):
    """u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = data[i,3]*np.cos(u)*np.sin(v) + data[i,0]
    y = data[i,3]*np.sin(u)*np.sin(v) + data[i,1]
    z = data[i,3]*np.cos(v) + data[i,2]
    ax.plot_surface(x, y, z, color='b',alpha=data[i,4])"""
    #scatrer and change the size of the points based on the radius
    ax.scatter(data[i,0] , data[i,1], data[i,2], s=data[i,3]*10000,alpha=data[i,4])

plt.show()

#plot the hitorgram of the opacity values as well as the radius values
data = np.loadtxt('output2.txt')
plt.hist(np.log(data[:,3]), bins=100, alpha=0.75)
plt.title('Histogram of the radius values')
plt.show()
plt.hist(np.log(data[:,4]), bins=100, alpha=0.75)
plt.title('Histogram of the opacity values')
plt.show()

#scatter plot of the radius values vs the opacity values
plt.scatter(data[:,3], data[:,4],alpha=0.5)
plt.xlabel('Radius')
plt.ylabel('opacity')
plt.semilogx()
plt.semilogy()
plt.show()

