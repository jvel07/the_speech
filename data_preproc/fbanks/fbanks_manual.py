import numpy

# converting Hertz (f) to Mel (m), see the formular above
def hertz_to_mels(f):
    return (2595 * numpy.log10(1 + f / 700.))


# converting Mel (m) to Hertz (f), see the formular above
def mels_to_hertz(m):
    return 700. * (10 ** (m / 2595.) - 1)


# This function constructs these Mel triangular filters
# It is taken from here:
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
def compute_mel_filters(sample_rate, high_freq=8000.,
                        low_freq=0., NFFT=512, nfilt=24):
    low_freq_mel = hertz_to_mels(low_freq)  # Convert Hz to Mel
    high_freq_mel = hertz_to_mels(high_freq)  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = mels_to_hertz(mel_points)  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank


# call the function with our parameters and compute the mel-filters
fbank = compute_mel_filters(rate, high_freq=rate / 2., low_freq=0.,
                            NFFT=size_of_fft, nfilt=num_mel_filters)