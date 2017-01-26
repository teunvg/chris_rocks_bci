import socket
import time

# Connect to Cybathlon
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error:
    print 'Failed to create socket'
    sys.exit()

host = '131.174.105.192'  #'localhost'
port = 5555
connection = (s, (host, port))

def sendCommand(connection, player, cmd):
    cmds = ['SPEED', 'JUMP', 'ROLL']
    msg = player * 10 + cmds.index(cmd) + 1
    connection[0].sendto(chr(msg), connection[1])

while True:  
    sendCommand(connection,1,'SPEED')
    print('speed')
    time.sleep(2)