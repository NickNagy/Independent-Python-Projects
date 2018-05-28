import re
from scipy.fftpack import fft, fftfreq
import numpy as np

FILE_NAME = 'eminor.dat'
READS_PER_SEC = 1

fileFreqSequence = [] # to use for sequential frequency storage

# modified FFT method from:
# https://stackoverflow.com/questions/50556705/accurate-method-for-most-prominent-frequencies-of-an-audio-signal/50557340#50557340
def fft_ups(data, rate, upsampling_factor=100):
    """fft with upsampling

    allows precise fft analysis by lengthening the
    target length.
    data:
        the data on which fft is performed
    upsampling_factor[opt]:
        the factor by which the data length is multiplied
    returns tuple (frequencies, amplitudes). amps are coeffed. mirror
        data is cut
    """
    def get_target():
        "as numpy's fft works best with powers of 2, find the smallest"
        "power near upsampling_factor"""
        for i in range(50):
            if 2**i > len(data)*upsampling_factor:
                return 2**(i-1)

    "coeff corrects amplitude"
    coeff = 2/len(data)
    target_length = get_target()

    "the half-length trick is to get rid of mirrored data"
    xf = fftfreq(target_length, 1/rate)[:target_length//2]
    yf = coeff*fft(data, target_length)[:target_length//2]
    return xf, yf

soundFile = open(FILE_NAME) # text file of WAV file data

firstLine = soundFile.readline()
fsString = firstLine[14:] # cuts off "; Sample Rate "
fs = int(fsString) # sample rate of file

secondLine = soundFile.readline()
channelsString = secondLine[11:] # cuts off "; Channels "
numChannels = int(channelsString)

dataList = []

# store second column (channel input) of DAT file in a list
for dataLine in soundFile:
    dataLine = " ".join(re.split("\s+", dataLine.strip(), flags = re.UNICODE))
    dataLineList = dataLine.split(" ")
    dataLineVal = 0
    if numChannels == 2:
        dataLineLeftVal = float(dataLineList[2])
        dataLineRightVal = float(dataLineList[len(dataLineList)-1])
        dataLineVal = 0.5*(dataLineLeftVal + dataLineRightVal) # take average of both channels
    else: # numChannels == 1
        dataLineVal = float(dataLineList[len(dataLineList)-1])
    dataList.append(dataLineVal)

index = 0

# store frequencies in fileFreqSequence while reading through file
while index < len(dataList):
    freq, amp = fft_ups(dataList[index:index+(fs*READS_PER_SEC)], fs)
    freqProminent = freq[np.argmax(amp)] # freq[index of max amplitude]
    fileFreqSequence.append(freqProminent)
    index += fs*READS_PER_SEC

print(fileFreqSequence)
