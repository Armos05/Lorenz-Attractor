
## Figure 1
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Lorenz paramters and initial conditions.
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05
# Maximum time point and total number of time points.
tmax, n = 100, 20000
def lorenz(t, X, sigma, beta, rho):
u, v, w = X
up = -sigma*(u - v)
vp = rho*u - v - u*w
wp = -beta*w + u*v
return up, vp, wp
# Integrate the Lorenz equations.
soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
dense_output=True)
# Interpolate solution onto the time grid, t.
t = np.linspace(0, tmax, n)
x, y, z = soln.sol(t)
fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI))ax = fig.gca(projection='3d')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
s = 10
cmap = plt.cm.winter
for i in range(0,n-s,s):
ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()


## Figure 2
cmap = plt.cm.winter
s = 10
for i in range(0,n-s,s):
plt.scatter(x[i:i+s+1], y[i:i+s+1],s = 1, color=cmap(i/n), alpha=0.4)
#plt.title("3D Lorenz System")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
for i in range(0,n-s,s):
plt.scatter(x[i:i+s+1], z[i:i+s+1], s = 1 ,color=cmap(i/n), alpha=0.4)
#plt.title("3D Lorenz System")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.show()for i in range(0,n-s,s):
plt.scatter(y[i:i+s+1], z[i:i+s+1], s = 1 ,color=cmap(i/n), alpha=0.4)
#plt.title("3D Lorenz System")
plt.xlabel("Y-axis")
plt.ylabel("Z-axis")
plt.show()

## Figure 3
dr = 0.05 # parameter step size
r = np.arange(1, 200, dr) # parameter range
dt = 0.001 # time step
t = np.arange(0, 10, dt) # time range
xs = np.empty(len(t) + 1)
ys = np.empty(len(t) + 1)
zs = np.empty(len(t) + 1)
# initial values x0,y0,z0 for the system
xs[0], ys[0], zs[0] = (1, 1, 1)
r_maxes = []
z_maxes = []
for R in r:
for i in range(len(t)):
# approximate numerical solutions to system
X = xs[i], ys[i], zs[j]
x_dot, y_dot, z_dot = lorenz(t,X,10,8/3 R)
xs[i + 1] = xs[i] + (x_dot * dt)ys[i + 1] = ys[i] + (y_dot * dt)
zs[i + 1] = zs[i] + (z_dot * dt)
# calculate and save the peak values of the z solution
for i in range(1, len(zs) - 1):
# save the local maxima
if zs[i - 1] < zs[i] and zs[i] > zs[i + 1]:
r_maxes.append(R)
z_maxes.append(zs[i])
xs[0], ys[0], zs[0] = xs[i], ys[i], zs[i]
plt.figure(figsize=(10, 5))
cmap = plt.cm.winter
plt.scatter(r_maxes, z_maxes, color = cmap(random.randint(0,200)) , s=0.5, alpha=0.2)
plt.xlabel("r")
plt.ylabel("z(t)")

## Figure 7
from scipy.signal import find_peaks
peaks, _ = find_peaks(z, height=0)# Plot Tent
zn = z[peaks[:-1]]
zn1 = z[peaks[1:]]
plt.scatter(zn,zn1)
plt.xlabel('Z_n')
plt.ylabel('Z_(n+1)')

