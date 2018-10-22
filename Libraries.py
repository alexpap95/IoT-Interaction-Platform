import numpy as np 
import quaternion

class Nodes:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def rotate(self, q):
        q1=np.quaternion(q[0],q[1],q[2],q[3])
        q2=np.conjugate(q1)
        v=np.quaternion(0,self.x,self.z,self.y)
        rot=(q1*v)*q2
        return Nodes(rot.x,rot.z,rot.y)

    def project(self, win_width, win_height, fov, viewer_distance):
        transformation = [550,220]
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + win_width  / 1.6
        y = -self.y * factor + win_height / 1.6
        return Nodes(x - transformation[0], y - transformation[1], self.z)

