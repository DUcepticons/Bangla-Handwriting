# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:51:43 2020

@author: akash
"""

from time import sleep
import serial

def single_line_send(a):
    sleep(1)
    print ('Sending: ' + a)
    data = a + '\n'
    ser.write(data.encode())
    sleep(1)
    print(ser.readline().strip())
    
def file_send(path):
    f = open(path,'r')
    for line in f:
        l = line.strip() # Strip all EOL characters for consistency
        print('Sending: ' + l)
        data = l + '\n'
        ser.write(data.encode()) # Send g-code block to grbl
        grbl_out = ser.readline()
        print(grbl_out.strip())

try:
    #setup
    ser = serial.Serial('COM3', 115200) # Establish the connection on a specific port
    sleep(2)   # Wait for grbl to initialize 
    ser.flushInput()  # Flush startup text in serial input
    ser.write(b"\r\n\r\n") #needed to be make it ready     ##  b"data" and "data".encode() are same
    sleep(1)
    
    '''
    single_line_send('M03 S60')
    single_line_send('M05')
    single_line_send('M03 S60')
    single_line_send('M05')
    '''
    
    file_send("Generated gcodes\\j.gcode")
    ser.close()
    
except Exception as e:
    print(e)
    ser.close()





'''


import serial
import time

# Open grbl serial port
s = serial.Serial('/dev/tty.usbmodem1811',115200)

# Open g-code file
f = open('grbl.gcode','r');

# Wake up grbl
s.write("\r\n\r\n")
time.sleep(2)   # Wait for grbl to initialize 
s.flushInput()  # Flush startup text in serial input

# Stream g-code to grbl
for line in f:
    l = line.strip() # Strip all EOL characters for consistency
    print 'Sending: ' + l,
    s.write(l + '\n') # Send g-code block to grbl
    grbl_out = s.readline() # Wait for grbl response with carriage return
    print ' : ' + grbl_out.strip()

# Wait here until grbl is finished to close serial port and file.
raw_input("  Press <Enter> to exit and disable grbl.") 

# Close file and serial port
f.close()
s.close()    

'''