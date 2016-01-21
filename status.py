from time import sleep
import sys

for i in range(101):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*(i/5), i))
    sys.stdout.flush()
    sleep(0.25)
