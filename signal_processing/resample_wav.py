import librosa
from common import util
import soundfile

work_dir = 'C:/Users/Win10/PycharmProjects/the_speech/audio/wav_anon_75_225/'
out_dir = 'C:/Users/Win10/PycharmProjects/the_speech/audio/resampled_anon_75_225/'
orig_audios = util.just_original_75()

for i in orig_audios:
    y, sr = librosa.load(work_dir + i, sr=16000)
    y_8k = librosa.resample(y, sr, 8000)
    soundfile.write(out_dir + i, y_8k, 8000, 'PCM_16')
    print("Resampled:", i)