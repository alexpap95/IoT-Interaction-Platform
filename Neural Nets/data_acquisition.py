import paho.mqtt.client as mqtt
import time
import json
from fusion import Fusion
## Change to Sensor Address (Without :)
sensor_mac = "B0B448C44883"

# The callback for when the client receives a CONNACK response from the server.
def s16(value):
	return -(value & 0x8000) | (value & 0x7fff)
		
def on_connect(client, userdata, flags, rc):
    global p
	print("Connected with result code "+str(rc))
	# Subscribing in on_connect() means that if we lose the connection and
	# reconnect then subscriptions will be renewed.
	client.subscribe("wsn/data/movement/"+sensor_mac)
    p=Process(target=wait_space)
    p.daemon = True
    p.start()

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
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
    p.stop()
    p.join()
    print("Disconnected with result code "+str(rc))

def wait_space():
    i=15
    while (True):
        if (i==15):
            input ("Press Enter...")
        fuse.write_to_file()
        i-=1
        if (i==0):
            i=15

client = mqtt.Client("wsncontroller@12345678")
client.username_pw_set("wsncontroller", "wSnC0ct1r")
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.connect("192.168.0.23", 1883, 60)
client.loop_forever(0.01)
