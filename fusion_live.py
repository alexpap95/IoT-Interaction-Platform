from math import sqrt, radians
from test_live import generate_data
import numpy as np

class Fusion(object):
    magmax = [-1000, -1000, -1000]
    magmin = [1000, 1000, 1000]                    
    def __init__(self, timediff=None):
        self.magbias = (0, 0, 0)            # local magnetic bias factors: set from calibration`               
        self.deltat = DeltaT(timediff)      # Time between updates
        self.q = [1.0, 0.0, 0.0, 0.0]       # vector to hold quaternion
        GyroMeasError = radians(40)         # Original code indicates this leads to a 2 sec response time
        self.beta = sqrt(3.0 / 4.0) * GyroMeasError  # compute beta (see README)
        self.c=15
        self.T=0
        self.gesture=np.zeros((15,10))
        
    def calibrate(self, mag):
        magxyz = tuple(mag)
        for x in range(3):
            self.magmax[x] = max(self.magmax[x], magxyz[x])
            self.magmin[x] = min(self.magmin[x], magxyz[x])
        self.magbias = tuple(map(lambda a, b: (a +b)/2, self.magmin, self.magmax))
        self.avg_del = tuple(map(lambda a, b: (a -b)/2, self.magmax, self.magmin))
        self.avg_delta = (self.avg_del[0] + self.avg_del[1] + self.avg_del[2]) / 3
        self.scale = (self.avg_delta/self.avg_del[0] if self.avg_del[0] else 0, self.avg_delta/self.avg_del[1] if self.avg_del[1] else 0,
         self.avg_delta/self.avg_del[2] if self.avg_del[2] else 0)

    def set_centre(self,qq):
        quat = tuple(qq)
        self.cen = [quat[0], quat[1], quat[2], quat[3]]   # vector to hold centre
        
    def update(self, accel, gyro, mag, ts=None):     # 3-tuples (x, y, z) for accel, gyro and mag data
        my, mx, mz = ((mag[x] - self.magbias[x])*self.scale[x] for x in range(3)) # Units irrelevant (normalised)
        ax, ay, az = accel                  # Units irrelevant (normalised)
        gx, gy, gz = (radians(x) for x in gyro)  # Units deg/s
        q1, q2, q3, q4 = (self.q[x] for x in range(4))
        # Auxiliary variables to avoid repeated arithmetic
        _2q1 = 2 * q1
        _2q2 = 2 * q2
        _2q3 = 2 * q3
        _2q4 = 2 * q4
        _2q1q3 = 2 * q1 * q3
        _2q3q4 = 2 * q3 * q4
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q1q4 = q1 * q4
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q2q4 = q2 * q4
        q3q3 = q3 * q3
        q3q4 = q3 * q4
        q4q4 = q4 * q4

        
        mz=-mz

        # Normalise accelerometer measurement
        norm = sqrt(ax * ax + ay * ay + az * az)
        if (norm == 0):
            return # handle NaN
        norm = 1 / norm                     # use reciprocal for division
        ax *= norm
        ay *= norm
        az *= norm

        # Normalise magnetometer measurement
        norm = sqrt(mx * mx + my * my + mz * mz)
        if (norm == 0):
            return                          # handle NaN
        norm = 1 / norm                     # use reciprocal for division
        mx *= norm
        my *= norm
        mz *= norm

        # Reference direction of Earth's magnetic field
        _2q1mx = 2 * q1 * mx
        _2q1my = 2 * q1 * my
        _2q1mz = 2 * q1 * mz
        _2q2mx = 2 * q2 * mx
        hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
        hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
        _2bx = sqrt(hx * hx + hy * hy)
        _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
        _4bx = 2 * _2bx
        _4bz = 2 * _2bz

        # Gradient descent algorithm corrective step
        s1 = (-_2q3 * (2 * q2q4 - _2q1q3 - ax) + _2q2 * (2 * q1q2 + _2q3q4 - ay) - _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4)
             + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
             + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s2 = (_2q4 * (2 * q2q4 - _2q1q3 - ax) + _2q1 * (2 * q1q2 + _2q3q4 - ay) - 4 * q2 * (1 - 2 * q2q2 - 2 * q3q3 - az)
             + _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4)
             + _2bz * (q1q2 + q3q4) - my) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s3 = (-_2q1 * (2 * q2q4 - _2q1q3 - ax) + _2q4 * (2 * q1q2 + _2q3q4 - ay) - 4 * q3 * (1 - 2 * q2q2 - 2 * q3q3 - az)
             + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx)
             + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
             + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        s4 = (_2q2 * (2 * q2q4 - _2q1q3 - ax) + _2q3 * (2 * q1q2 + _2q3q4 - ay) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4)
              + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my)
              + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz))

        norm = 1 / sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)    # normalise step magnitude
        s1 *= norm
        s2 *= norm
        s3 *= norm
        s4 *= norm

        # Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        # Integrate to yield quaternion
        deltat = self.deltat(ts)
        q1 += qDot1 * deltat
        q2 += qDot2 * deltat
        q3 += qDot3 * deltat
        q4 += qDot4 * deltat
        norm = 1 / sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)    # normalise quaternion
        
        self.q = q1 * norm, q2 * norm, q3 * norm, q4 * norm
        
        cq1 = self.q[0]*self.cen[0]-self.q[1]*self.cen[1]-self.q[2]*self.cen[2]-self.q[3]*self.cen[3]
        cq2 = self.q[0]*self.cen[1]+self.q[1]*self.cen[0]-self.q[2]*self.cen[3]+self.q[3]*self.cen[2]
        cq3 = self.q[0]*self.cen[2]+self.q[1]*self.cen[3]+self.q[2]*self.cen[0]-self.q[3]*self.cen[1]
        cq4 = self.q[0]*self.cen[3]-self.q[1]*self.cen[2]+self.q[2]*self.cen[1]+self.q[3]*self.cen[0]
        
        normc = 1 / sqrt(cq1 * cq1 + cq2 * cq2 + cq3 * cq3 + cq4 * cq4)
        self.cq = cq1 * normc, cq2 * normc, cq3 * normc, cq4 * normc
      
        if (self.c<15 and self.c>0):
            self.T+=deltat*1000
        else:
            self.T=0
            self.c=15
        
        accx=(accel[0]/9.8)*1000
        accy=(accel[1]/9.8)*1000
        accz=(accel[2]/9.8)*1000
        gyrx=gyro[0]*1000
        gyry=gyro[1]*1000
        gyrz=gyro[2]*1000
        
        self.data=[accx,accy,accz,gyrx,gyry,gyrz,cq1*1000,cq2*1000,cq3*1000,cq4*1000]
        self.gesture = np.delete(self.gesture, 0, 0)
        self.gesture = np.vstack((self.gesture, self.data))
        generate_data(self.gesture)     

class DeltaT():
    def __init__(self, timediff):
        if timediff is None:
            self.expect_ts = False
            raise ValueError('You must define a timediff function')
        else:
            self.expect_ts = True
            self.timediff = timediff
        self.start_time = None

    def __call__(self, ts):
        if self.expect_ts:
            if ts is None:
                raise ValueError('Timestamp expected but not supplied.')
        else:
            raise RuntimeError('Provide timestamps and a timediff function')
        # ts is now valid
        if self.start_time is None:  # 1st call: self.start_time is invalid
            self.start_time = ts
            return 0.0001  # 100μs notional delay. 1st reading is invalid in any case

        dt = self.timediff(ts, self.start_time)
        self.start_time = ts
        return dt
