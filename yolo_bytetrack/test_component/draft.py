import time
import os


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
i = 0
while True:
    i += 1
    clear_screen()
    print('alo', i)
    time.sleep(1)
    print()
