import paho.mqtt.client as mqtt
import time
## Change to Sensor Address (Without :)
sensor_mac = "B0B448C44883"
i=1000
accmax = [-2, -2, -2]
accmin = [2, 2, 2] 
acc = [0,0,0]
# The callback for when the client receives a CONNACK response from the server.
def s16(value):
		return -(value & 0x8000) | (value & 0x7fff)
		
def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))
	time.sleep(0.5)
	print("Keep the sensor completely steady")
	open('accdata', 'w').close()
	time.sleep(0.5)
	# Subscribing in on_connect() means that if we lose the connection and
	# reconnect then subscriptions will be renewed.
	client.subscribe("wsn/data/movement/"+sensor_mac)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
	global i
	temp=str(msg.payload)[2:55]
	valueByte = bytearray.fromhex(temp)
	scaleacc = 4096
	accx = (s16((valueByte[7]<<8) + valueByte[6]))/scaleacc
	accy = (s16((valueByte[9]<<8) + valueByte[8]))/scaleacc
	accz = (s16((valueByte[11]<<8) + valueByte[10]))/scaleacc-1
	if (abs(accx-acc[0]) > 0.15):
		acc[0]=accx
	if (abs(accy-acc[1]) > 0.15):
		acc[1]=accy
	if (abs(accz-acc[2]) > 0.15):
		acc[2]=accz
	for x in range(3):
	    accmax[x] = max(accmax[x], acc[x])
	    accmin[x] = min(accmin[x], acc[x])
	accbias = tuple(map(lambda a, b: (a +b)/2, accmin, accmax))
	accabs = tuple(map(lambda a, b: (a -b)/2, accmax, accmin))
	with open('accdata', 'a') as f:
		f.write(str(acc) + '\n')
		print(str(acc))
		i -= 1
		if (i<0):
			f.write(str(accbias) + '\n')
			f.write(str(accabs) + '\n')
			print("The bias is" + str(accbias))
			print("The difference is" + str(accabs))
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
