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
        wav_resampled = resample_wav('/media/jose/hk-data/PycharmProjects/the_speech/audio/wav16k_split_long/'+wav, target_sample_rate=target_sample_rate)
        save_wav(wav_resampled, out_dir, target_sample_rate, subtype)
    print("Finished resampling {} wavs in {}".format(len(list_wavs), os.path.dirname(wav)))
    print("Resampling details:\n all wavs resampled to {}; with {}".format(target_sample_rate, subtype))
    print("Resampled audios saved to:", out_dir)


wavs_dir = '/media/jose/hk-data/PycharmProjects/the_speech/audio/wav16k_split_long'
list_wavs = util.read_files_from_dir(wavs_dir)

make_resample(list_wavs, out_dir='/media/jose/hk-data/PycharmProjects/the_speech/audio/bea_wav8k',
              target_sample_rate=8000, subtype='PCM_16')
