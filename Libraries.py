import math, time, pygame, random, string
import numpy as np 
import quaternion
from numpy.linalg import inv
from operator import itemgetter

#initialize the list contaenes the order which the angles have to occurr in order to recognize the patern (up, down, right,left)
def initialize_1st_sequence(sequence_angle,first_sequence):
    for x in range (0, len(sequence_angle)):
            first_sequence.insert(0,[0,0,0,0])
    #also make sure to clear the sequence list after every try
    return first_sequence[0:len(sequence_angle)]


'''MECHANISM FOR RECOGNIZE PATERN'''
#function for recognize sequence: pitch+, pitch-, roll+, roll-
def patern_recognisionPprR(pitch,roll,sequence_angle,first_sequence,succed_patern):
    #until the 1st angle recognised in first_sequence
    for this_angle in range (0, len(sequence_angle)):
        if pitch > sequence_angle[this_angle] and first_sequence[this_angle] == [0,0,0,0]:
            first_sequence[this_angle].append(1)
            del first_sequence[this_angle][0]

        elif pitch < -sequence_angle[this_angle] and first_sequence[this_angle] == [0,0,0,1]:
            first_sequence[this_angle].append(1)
            del first_sequence[this_angle][0]

        elif roll < -sequence_angle[this_angle] and first_sequence[this_angle] == [0,0,1,1]:
            first_sequence[this_angle].append(1)
            del first_sequence[this_angle][0]

        elif roll > sequence_angle[this_angle] and first_sequence[this_angle] == [0,1,1,1]:
            succed_patern.append(this_angle)
            first_sequence[this_angle] = [0,0,0,0]

    return first_sequence, succed_patern


def fire_the_countdown(pitch):
    if pitch > 15:
        #start countdown
        clock_state = 1
        return clock_state
    else:
        return 0
        
def get_current_time():
    time_start = time.time()
    return time_start

def get_clock(time_start, time_window):
        #calculate the time passed until now in seconds
        seconds = int(time.time() - time_start)
        #check if the countdown is over
        if seconds == time_window:
            #reset the countdown
            return 0, 7
        else:
            return 1, 7-seconds

#4
class Nodes:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def rotate(self, q):
        q1=np.quaternion(q[0],q[1],q[2],q[3])
        q2=np.conjugate(q1)
        v=np.quaternion(0,self.x,self.z,self.y)
        rot=(q1*v)*q2
        return Nodes(rot.x,rot.z,rot.y)

    # def move(self, quat, accel):
    #     q=np.quaternion(quat[0],quat[1],quat[2],quat[3])
    #     R=quaternion.as_rotation_matrix(q)
    #     invR=inv(R)
    #     acc=np.array(accel).reshape(3,1)
    #     absolute_acc=np.dot(invR,acc)
    #     self.x += absolute_acc[0]/2
    #     self.y += absolute_acc[1]/2
    #     self.z += absolute_acc[2]/2
    #     return Nodes(self.x,self.y,self.z)

    def project(self, win_width, win_height, fov, viewer_distance):
        transformation = [550,220]
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + win_width  / 1.6
        y = -self.y * factor + win_height / 1.6
        return Nodes(x - transformation[0], y - transformation[1], self.z)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

