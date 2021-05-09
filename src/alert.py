from playsound import playsound
import os


# Plays the beep sound t
def play():
    playsound(os.path.join('src', 'beep.mp3'))
