#!/usr/bin/env python3

import wave
import json
import sys
import librosa

import soundfile as sf


from itertools import islice
from multiprocessing.dummy import Pool
import threading
from vosk import Model, KaldiRecognizer

import queue

import pyaudio, time, asyncio, json

filename= 'audio_files/lecture1.wav'
file=wave.open(filename,mode='rb')
rate=file.getframerate()

# model = Model(lang='ru')
model = Model('../data/vosk-model-ru-0.42')
rec = KaldiRecognizer(model, rate)
rec.SetWords(True)



proc_time=0.1
chunk=int(proc_time*rate)

while True:
    data=file.readframes(chunk)
    if not data:
        break


    if rec.AcceptWaveform(data):
        res=json.loads(rec.Result())
        print(res)


