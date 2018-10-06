import paho.mqtt.client as mqtt
import time
## Change to Sensor Address (Without :)
sensor_mac = "B0B448C44883"
i=300
# The callback for when the client receives a CONNACK response from the server.
def s16(value):
		return -(value & 0x8000) | (value & 0x7fff)
		
def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))
	time.sleep(1)
	print("Start rotating around each axis")
	open('mpudata', 'w').close()
	time.sleep(1)
	# Subscribing in on_connect() means that if we lose the connection and
	# reconnect then subscriptions will be renewed.
	client.subscribe("wsn/data/movement/"+sensor_mac)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
	global i
	temp=str(msg.payload)[2:55]
	valueByte = bytearray.fromhex(temp)
	scale = 32760/49120
	magx = (s16((valueByte[13]<<8) + valueByte[12]))/scale
	magy = (s16((valueByte[15]<<8) + valueByte[14]))/scale
	magz = (s16((valueByte[17]<<8) + valueByte[16]))/scale
	mag = [magx,magy,magz]
	with open('mpudata', 'a') as f:
		f.write(str(mag) + '\n')
		print(str(mag))
		i -= 1
		if (i<0):
			client.disconnect()
		f.close()

def on_disconnect(client,userdata,rc):
	print("Calibration ended. Disconnected with result code "+str(rc))

client = mqtt.Client("wsncontroller@12345678")
client.username_pw_set("wsncontroller", "wSnC0ct1r")
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.connect("192.168.0.23", 1883, 60)
client.loop_forever(0.1)
