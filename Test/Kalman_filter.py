from pykalman import KalmanFilter
import numpy as np

#import matplotlib.pyplot as plt
# time step
dt = 0.05

# transition_matrix  
A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# observation_matrix   
H = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# In[6]:

rp = 0.002 # Variance of Acc measurement
R = np.matrix([[rp, 0.0, 0.0],
               [0.0, rp, 0.0],
               [0.0, 0.0, rp]])

# In[8]:

sa = 0.1
G = np.matrix([[1/2.0*dt**2],
               [1/2.0*dt**2],
               [1/2.0*dt**2],
               [dt],
               [dt],
               [dt],
               [1.0],
               [1.0],
               [1.0]])
Q = G*G.T*sa**2
# initial_state_mean
x = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01171875, -0.002685546875, 0.260223388671875]).T

# initial_state_covariance
P0 = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rp, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rp, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rp]])

n_dim_state = 9
new_state_means = np.zeros(n_dim_state)
new_state_covariances = np.zeros(n_dim_state, n_dim_state)
last_state_means = np.zeros(n_dim_state)
last_state_covariances = np.zeros(n_dim_state, n_dim_state)

kf = KalmanFilter(transition_matrices = A, 
                  observation_matrices = H, 
                  transition_covariance = Q, 
                  observation_covariance = R, 
                  initial_state_mean = X0, 
                  initial_state_covariance = P0)

if t == 0:
    last_state_means = X0
    last_state_covariances = P0
else:
    new_state_means, new_state_covariances = (
    kf.filter_update(
        last_state_means,
        last_state_covariances,
        Acceleration
    )
)

print (last_state_means)


# f, axarr = plt.subplots(3, sharex=True)

# axarr[0].plot(Time, AccX_Value, label="Input AccX")
# axarr[0].plot(Time, filtered_state_means[:, 2], "r-", label="Estimated AccX")
# axarr[0].set_title('Acceleration X')
# axarr[0].grid()
# axarr[0].legend()
# axarr[0].set_ylim([-4, 4])

# axarr[1].plot(Time, RefVelX, label="Reference VelX")
# axarr[1].plot(Time, filtered_state_means[:, 1], "r-", label="Estimated VelX")
# axarr[1].set_title('Velocity X')
# axarr[1].grid()
# axarr[1].legend()
# axarr[1].set_ylim([-1, 20])

# axarr[2].plot(Time, RefPosX, label="Reference PosX")
# axarr[2].plot(Time, filtered_state_means[:, 0], "r-", label="Estimated PosX")
# axarr[2].set_title('Position X')
# axarr[2].grid()
# axarr[2].legend()
# axarr[2].set_ylim([-10, 1000])

# plt.show()