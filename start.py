# -*- coding: utf-8 -*-
import cPickle
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from sklearn import preprocessing
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.mixture import GMM
import os
import thread
import time


import python_speech_features as mfcc
from scipy.io.wavfile import read


def get_MFCC(file):
    sr, audio = read(file)
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy = False)
    features = preprocessing.scale(features)
    return features

def start_rec():
    import pyaudio
    import wave

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "file.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print "recording..."
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print "finished recording"

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def calculate_delta(array):

    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):

    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))
    return combined




def create_model():

    alex = ['ak.wav', 'ak2.wav', 'ak_cut.wav']
    kvar = ['kv.wav', 'kv2.wav', 'kv_cut.wav']

    features = np.asarray(())
    for i in alex:
        sr, audio = read(i)

        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))


    gmm = GMM(n_components=16, n_iter=200, covariance_type='diag', n_init=3)
    gmm.fit(features)

    picklefile = 'alex' + ".gmm"
    cPickle.dump(gmm, open(picklefile, 'w'))
    print '+ modeling completed for speaker:', picklefile, " with data point = ", features.shape


    features = np.asarray(())
    for i in kvar:
        sr, audio = read(i)

        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))


    gmm = GMM(n_components=16, n_iter=200, covariance_type='diag', n_init=3)
    gmm.fit(features)

    picklefile = 'kvar' + ".gmm"
    cPickle.dump(gmm, open(picklefile, 'w'))
    print '+ modeling completed for speaker:', picklefile, " with data point = ", features.shape







def get_voice():
    models = [cPickle.load(open(fname, 'r')) for fname in ['alex.gmm', 'kvar.gmm']]

    sr, audio = read('file.wav')
    vector = extract_features(audio, sr)
    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)

    if winner == 1:
        print('kvar')
        thread.start_new_thread( start_play, ('vera_kv.mp3', ))
    else:
        print('alex')
        thread.start_new_thread( start_play, ('vera_ak.mp3', ))



def start_play(file):
    os.system('ffplay ' + file)

print('start')
time.sleep(4)
thread.start_new_thread( start_play, ('kv_5.wav', ))
start_rec()
get_voice()


time.sleep(6)

thread.start_new_thread( start_play, ('ak_5.wav', ))
start_rec()
get_voice()


