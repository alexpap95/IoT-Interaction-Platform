import paho.mqtt.client as mqtt
import time
from fusion_acq import Fusion
import threading
import json


## Change to Sensor Address (Without :)
sensor_mac = "B0B448C44883"
fuse = Fusion(lambda start, end: start-end)
# The callback for when the client receives a CONNACK response from the server.

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
            
def s16(value):
	return -(value & 0x8000) | (value & 0x7fff)
		
def main():
    def on_connect(client, userdata, flags, rc):
        global p
        print("Connected with result code "+str(rc))
    	# Subscribing in on_connect() means that if we lose the connection and
    	# reconnect then subscriptions will be renewed.
        client.subscribe("wsn/data/movement/"+sensor_mac)
    
    # The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
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
        accel = (accx,accy,accz)
        gyro= (gyrox,gyroy,gyroz)
        mag = (magx,magy,magz)
        fuse.update(accel,gyro,mag,time.time())
    
    def on_disconnect(client,userdata,rc):
        global p
        t.stop()
        t.join()
        print("Disconnected with result code "+str(rc))
        
    client = mqtt.Client("wsncontroller@12345678")
    client.username_pw_set("wsncontroller", "wSnC0ct1r")
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect("192.168.1.8", 1883, 60)
    client.loop_forever(0.01)

class myThread(threading.Thread):
    def __init__(self,fuse):
        super(myThread, self).__init__()
        self.fuse=fuse
    def run(self):
        time.sleep(1)
#        for n in range(3600):
#            time.sleep(0.1)
#            self.fuse.write_to_file()
#        print("done")
        i=15
        N=1
        while (True):
            time.sleep(0.1)
            if (i==15):
                input ("Rep " + str(N) + " - Press Enter...")
                print ("Ready")
                time.sleep(0.3)
                print ("GO")
                time.sleep(0.3)
            self.fuse.write_to_file()
            i-=1
            if (i==0):
                i=15
                N+=1

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
    t=myThread(fuse)
    t.daemon=True
    t.start()
    main()
