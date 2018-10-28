import paho.mqtt.client as mqtt
import json
import time, pygame
import numpy as np
from operator import itemgetter
from fusion import Fusion

black = (0,0,0)
red = (255,0,0)
green = (0,170,0)
sensor_mac = "B0B448C44883"
fuse = Fusion(lambda start, end: start-end)
oldpitch=[0]
oldroll=[0]
oldheading=[0]
samples=5
j=1
k=0
sumpitch=0
sumroll=0
sumheading=0


def gdata():
    # Return [[ax, ay, az], [gx, gy, gz], [mx, my, mz], timestamp]
    # from whatever the device supplies (in this case JSON)
    with open('mpudata', 'r') as f:
        line = f.readline()  # An app would do a blocking read of remote data
        while line:
            yield json.loads(line)  # Convert foreign data format.
            line = f.readline()  # Blocking read.
def init_quat():
    with open('centre', 'r') as f:
        line = f.readline()  # An app would do a blocking read of remote data
        while line:
            yield json.loads(line)  # Convert foreign data format.
            line = f.readline()  # Blocking read.
            
def main():
    global glove,client
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("wsn/data/movement/"+sensor_mac)
        
    def on_message(client, userdata, msg):
        cyber_glove(msg).set_acc(msg)

    def on_disconnect(client,userdata,rc):
        print("Disconnected with result code "+str(rc))
        pygame.quit()
    
    client = mqtt.Client("wsncontroller@12345678")
    client.username_pw_set("wsncontroller", "wSnC0ct1r")
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect("192.168.1.6", 1883, 60)
    client.loop_forever(0.01)

def s16(value):
        return -(value & 0x8000) | (value & 0x7fff)

class cyber_glove():
    def __init__(self,msg):
        self.msg=msg
        '''3d cube variables'''
        self.vertices = [
                    Nodes(-1,0.2,-1.4),
                    Nodes(1,0.2,-1.4),
                    Nodes(1,-0.2,-1.4),
                    Nodes(-1,-0.2,-1.4),
                    Nodes(-1,0.2,1.4),
                    Nodes(1,0.2,1.4),
                    Nodes(1,-0.2,1.4),
                    Nodes(-1,-0.2,1.4)]
        self.faces = [(0,1,2,3),(1,5,6,2),(5,4,7,6),(4,0,3,7),(0,4,5,1),(3,2,6,7)]
        self.colors = [(255,69,0),(255,165,0),(0,255,0),(131,137,150),(0,0,255),(255,0,0)]
        '''graphics variables'''
        pygame.init()
        self.screen = pygame.display.set_mode((740, 700))
        self.smallfont = pygame.font.SysFont(None,80)
        pygame.display.set_caption("Wearable IMU")

    def set_acc(self,msg):
        global fuse
        temp=str(msg.payload)[2:55]
        valueByte = bytearray.fromhex(temp)
        scaleacc = 16384
        scalegyr = 32768/250
        scalemag = 32760/49120
        accx = (s16((valueByte[7]<<8) + valueByte[6]))/scaleacc
        accy = (s16((valueByte[9]<<8) + valueByte[8]))/scaleacc
        accz = (s16((valueByte[11]<<8) + valueByte[10]))/scaleacc
        gyrox = (s16((valueByte[1]<<8) + valueByte[0]))/scalegyr
        gyroy = (s16((valueByte[3]<<8) + valueByte[2]))/scalegyr
        gyroz = (s16((valueByte[5]<<8) + valueByte[4]))/scalegyr
        magx = (s16((valueByte[13]<<8) + valueByte[12]))/scalemag
        magy = (s16((valueByte[15]<<8) + valueByte[14]))/scalemag
        magz = (s16((valueByte[17]<<8) + valueByte[16]))/scalemag
        self.accel = (accx,accy,accz)
        self.gyro= (gyrox,gyroy,gyroz)
        self.mag = (magx,magy,magz)
        fuse.update(self.accel,self.gyro,self.mag,time.time())
        self.run()

    def monitor(self,x,y,digits, value,newx1 = 0, start = (0,0), stop = (0,0)):
        if digits == 3:
            self.screen_text = self.smallfont.render("PITCH", True, black)
            self.screen.blit(self.screen_text,[20,590])
            pygame.draw.rect(self.screen,black,(x,y,digits*70,88))
        elif digits == 4:
            self.screen_text = self.smallfont.render("ROLL", True, black)
            self.screen.blit(self.screen_text,[250,590])
            pygame.draw.rect(self.screen,black,(x,y,digits*70,88))
        elif digits == 5:
            digits=digits-1
            self.screen_text = self.smallfont.render("HEADING", True, black)
            self.screen.blit(self.screen_text,[460,590])
            pygame.draw.rect(self.screen,black,(x,y,digits*70,88))

        value = str(value)
        for i in range(0,digits):
            #six nodes tha compose a digit
            pointslist = [(x + 10 + newx1, y+ 7.5),
                      (x + 60 + newx1, y + 7.5),
                      (x + 10 + newx1, y + 45),
                      (x + 60 + newx1, y + 45),
                      (x + 10 + newx1, y + 82.5),
                      (x + 60 + newx1, y + 82.5)]
            #six vertices tha connect the nodes
            vertices = [(pointslist[0],pointslist[1]), # 0
                    (pointslist[2],pointslist[3]), # 1
                    (pointslist[4],pointslist[5]), # 2
                    (pointslist[0],pointslist[2]),# 3
                    (pointslist[1],pointslist[3]), # 4
                    (pointslist[2],pointslist[4]), # 5
                    (pointslist[3],pointslist[5])] # 6

            number = value[i:i+1]
            linescounter = -1
            if number == '0':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 1:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '1':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 4 or linescounter == 6:
                        start = n[0]
                        stop = n[1]
                        pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '2':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 3 or linescounter == 6:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '3':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 3 or linescounter == 5:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '4':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 0 or linescounter == 5 or linescounter == 2:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '5':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 4 or linescounter == 5:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '6':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 4 or linescounter == 0:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '7':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 3 or linescounter == 5 or linescounter == 1 or linescounter == 2:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '8':
                for n in vertices:
                    linescounter += 1
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '9':
                for n in vertices:
                    linescounter += 1
                    if linescounter == 5:
                        continue
                    start = n[0]
                    stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '-' :
                for n in vertices:
                    linescounter += 1
                    if linescounter == 1:
                        start = n[0]
                        stop = n[1]
                    pygame.draw.line(self.screen,red,start,stop,4)
            elif number == '.' :
                break

            newx1 += 60


    def rotatetor(self):
        global fuse
        '''SET THE NEW ANGLE POSITION OF THE CUBE - (FIND THE NEW PLACE OF THE ROTATED POINTS)'''
        # It will hold transformed vertices.
        t = []
        for v in self.vertices:
            # Rotate the point around z axis, around X axis roll,  around Y axis pitch
            r = v.rotate(fuse. cq)
            # Transform the point from 3D to 2D
            p = r.project(740, 700, 256, 2.8)
            # Put the point in the list of transformed vertices
            t.append(p)
        # Calculate the average Z values of each face.
        avg_z = []
        i = 0
        for f in self.faces:
            z = (t[f[0]].z + t[f[1]].z + t[f[2]].z + t[f[3]].z) / 4.0
            avg_z.append([i,z])
            i = i + 1
        # Draw the faces using the Painter's algorithm:
        # Distant faces are drawn before the closer ones.
        for tmp in sorted(avg_z,key=itemgetter(1),reverse=True):
            face_index = tmp[0]
            f = self.faces[face_index]
            pointlist = [(t[f[0]].x, t[f[0]].y), (t[f[1]].x, t[f[1]].y),
                         (t[f[1]].x, t[f[1]].y), (t[f[2]].x, t[f[2]].y),
                         (t[f[2]].x, t[f[2]].y), (t[f[3]].x, t[f[3]].y),
                         (t[f[3]].x, t[f[3]].y), (t[f[0]].x, t[f[0]].y)]
            #draw the new angular position of the cube
            pygame.draw.polygon(self.screen,self.colors[face_index],pointlist)

    def run(self):
        global fuse,oldpitch,oldroll,oldheading,client,j,k,sumpitch,sumroll,sumheading
        if (j<samples):
            oldpitch.append(fuse.pitch)
            oldroll.append(fuse.roll)
            oldheading.append(fuse.heading)
            sumpitch+=oldpitch[j]
            sumroll+=oldroll[j]
            sumheading+=oldheading[j]
            j+=1
        else:
            if (k==samples):
                k=0
            sumpitch-=oldpitch[k]
            sumroll-=oldroll[k]
            sumheading-=oldheading[k]
            oldpitch[k]=fuse.pitch
            oldroll[k]=fuse.roll
            oldheading[k]=fuse.heading
            sumpitch+=oldpitch[k]
            sumroll+=oldroll[k]
            sumheading+=oldheading[k]
            k+=1
        self.pitch=sumpitch//samples
        self.roll=sumroll//samples
        self.heading=sumheading//samples
        '''DRAW THE GRAPHICS'''
        #draw the graphical enviroment white
        pygame.draw.rect(self.screen,(255,255,255),(0,0,740,700))
        #draw the angular position of the cube
        self.rotatetor()
        #draw the values of pitch, roll and countdown clock on a digital lcd
        self.monitor(10,500,3,self.pitch)
        self.monitor(200,500,4,self.roll)
        self.monitor(450,500,5,self.heading)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                client.disconnect()
                
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
        transformation = [120,220]
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + win_width  / 1.6
        y = -self.y * factor + win_height / 1.6
        return Nodes(x - transformation[0], y - transformation[1], self.z)
                 
if __name__ == '__main__':
    i=200
    get_data=gdata()
    print('Intro')
    while (i>0):
        getmag=next(get_data)
        fuse.calibrate(getmag)
        i-=1    
    print('Cal done. Magnetometer bias vector:', fuse.magbias, fuse.scale)
    get_centre=init_quat()
    centre=next(get_centre)
    fuse.set_centre(centre)
    print("Centre initialised")
    main()
