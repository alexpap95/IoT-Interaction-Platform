import paho.mqtt.client as mqtt
import json
import time
from fusion_init import Fusion
import numpy as np
import quaternion 

sensor_mac = "B0B448C92601"
fuse = Fusion(lambda start, end: start-end)
counter=50

def gdata():
    # Return [[ax, ay, az], [gx, gy, gz], [mx, my, mz], timestamp]
    # from whatever the device supplies (in this case JSON)
    with open('mpudata', 'r') as f:
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
    
    client = mqtt.Client("wsncontroller@12345678")
    client.username_pw_set("wsncontroller", "wSnC0ct1r")
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect("192.168.1.103", 1883, 60)
    client.loop_forever(0.01)

def s16(value):
        return -(value & 0x8000) | (value & 0x7fff)

class cyber_glove():
    def __init__(self,msg):
        self.msg=msg
    def set_acc(self,msg):
        global fuse,counter,client
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
        if (counter>0):
            counter-=1
        else:
            centre=np.quaternion(fuse.q[0],fuse.q[1],fuse.q[2],fuse.q[3])
            inverse=np.conjugate(centre)
            with open('centre', 'w') as f:
                f.write(str([inverse.w,inverse.x,inverse.y,inverse.z]))
            client.disconnect() 
      
if __name__ == '__main__':
    i=200
    get_data=gdata()
    print('Intro')
    while (i>0):
        getmag=next(get_data)
        fuse.calibrate(getmag)
        i-=1
    print('Cal done. Magnetometer bias vector:', fuse.magbias, fuse.scale)
    main()
