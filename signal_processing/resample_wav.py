import librosa
from common import util
import soundfile
import os


def resample_wav(wav, target_sample_rate):
    y, sr = librosa.load(wav, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
    return y_resampled


def save_wav(data, out_dir, sample_rate, subtype='PCM_16'):
    soundfile.write(out_dir, data, sample_rate, subtype)


def make_resample(list_wavs, out_dir, target_sample_rate, subtype):
    for wav in list_wavs:
        wav_resampled = resample_wav(wav, target_sample_rate=target_sample_rate)
        save_wav(wav_resampled, wav, target_sample_rate, subtype)
    print("Finished resampling {} wavs in {}".format(len(list_wavs), os.path.dirname(wav)))
    print("Resampling details:\n all wavs resampled to {}; with {}".format(target_sample_rate, subtype))
    print("Resampled audios saved to:", out_dir)
