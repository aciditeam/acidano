"""
.. versionadded:: 1.1.0
   This demo depends on new features added to contourf3d.
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import math

import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

N = 100
D_min = -3
D_max = 3
step_size = float(D_max - D_min) / N

X = np.zeros((N,N))
Y = np.zeros((N,N))
Z = np.zeros((N,N))

for (i,x) in enumerate(np.arange(D_min, D_max, step_size)):
    for (j,y) in enumerate(np.arange(D_min, D_max, step_size)):
        X[i,j] = x
        Y[i,j] = y
        Z[i,j] = math.tanh((x+y)/2)

# n = 50
# xs = (D_max - D_min)*np.random.rand(n) + D_min
# ys = (D_max - D_min)*np.random.rand(n) + D_min
#
# zs = []
# for a, b in zip(xs, ys):
#     zs.append(math.tanh((a+b)/2))

xs = [-2, 0]
ys = [-2, 0]
zs = []
for a, b in zip(xs, ys):
    zs.append(math.tanh((a+b)/2))

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
ax.scatter(xs, ys, zs, c='r')

# cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
#
# ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
# ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)

ax.grid(b=False)
plt.axis('off')
plt.savefig('/Users/leo/Recherche/GitHub_Aciditeam/notes/SMC/Figures/manifold_single_point.pdf')
