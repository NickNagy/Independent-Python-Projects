import re
from scipy.fftpack import fft, rfft
import tensorflow as tf

FILE_NAME = '440Hz.dat'
READS_PER_SEC = 1

fileFreqSequence = [] # to use for sequential frequency storage

soundFile = open(FILE_NAME) # text file of WAV file data

firstLine = soundFile.readline()
fsString = firstLine[14:] # cuts off "; Sample Rate "
fs = int(fsString) # sample rate of file

secondLine = soundFile.readline()
channelsString = secondLine[11:] # cuts off "; Channels "
numChannels = int(channelsString)

dataList = []

dataLine = soundFile.readline()

# store second column (channel input) of DAT file in a list
if numChannels == 1:
    while dataLine is not None:
        dataLine = " ".join(re.split("\s+", dataLine.strip(), flags = re.UNICODE))
        dataList.append(int(dataLine[3]))
        dataLine = soundFile.readline()

index = 0

# store frequencies in fileFreqSequence while reading through file
while index < len(dataList):
    rawFTData = fft(dataList[index:index+(fs*READS_PER_SEC)])
    # TODO get accurate frequency from rawFTData
    # TODO remove data points outside of human hearing range
    freqProminent = rawFTData[0]
    fileFreqSequence.apppend(freqProminent)
    index += fs*READS_PER_SEC

# ----------------------------------------------------------------

# Below from Tensorflow: https://www.tensorflow.org/tutorials/recurrent
