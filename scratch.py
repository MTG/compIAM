

zeropadLen = params.Nfft - params.frmSize
zz = np.zeros((zeropadLen,), dtype='float32')
frameCounter = 0
bufferFrame = np.zeros((params.Nfft / 2 + 1,))
#logger.info('Reading audio file...')

#audio = ess.MonoLoader(filename=fname)()
audio, _ = librosa.load(fname, sampleRate=params.Fs)
#fft = ess.FFT(size=params.Nfft)  # this gives us a complex FFT
mag, phase = librosa.stft(audio)


#c2p = ess.CartesianToPolar()  # and this turns it into a pair (magnitude, phase)
pool = es.Pool()


#w = ess.Windowing(type="hamming")
scisig.get_window("hamming")

fTicks = params.fTicks
poolName = 'features.flux'
#logger.info('Extracting Onset functions...')
for frame in ess.FrameGenerator(audio, frameSize=params.frmSize, hopSize=params.hop):
    frmTime = params.hop / params.Fs * frameCounter + params.frmSize / (2.0 * params.Fs)
    zpFrame = np.hstack((frame, zz))
    mag, phase, = c2p(fft(w(zpFrame)))
    magFlux = mag - bufferFrame
    bufferFrame = np.copy(mag)  # Copying for the next iteration to compute flux
    for bands in range(params.numBands):
        chosenInd = (fTicks >= params.fBands[bands, 0]) & (fTicks <= params.fBands[bands, 1])
        magFluxBand = magFlux[chosenInd]
        magFluxBand = (magFluxBand + abs(magFluxBand)) / 2
        oFn = magFluxBand.sum()
        if (math.isnan(oFn)):
            print("NaN found here")
        pass
        pool.add(poolName + str(bands), oFn)
    pass
    pool.add('features.time', frmTime);
    frameCounter += 1
    if not np.mod(frameCounter, 10000):
        #logger.info(str(frameCounter) + '/' + str(audio.size / params.hop) + '...')
#logger.info('Total frames processed = ' + str(frameCounter))
timeStamps = es.array([pool['features.time']])
all_feat = timeStamps
for bands in range(params.numBands):
    feat_flux = es.array([pool[poolName + str(bands)]])
    all_feat = np.vstack((all_feat, feat_flux))
pass