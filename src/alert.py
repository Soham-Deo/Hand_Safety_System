from playsound import playsound
import os


def play():
    playsound(os.path.join('src', 'beep.mp3'))
