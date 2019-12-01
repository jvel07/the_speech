from specAugment import spec_augment_tensorflow
from common import util

original_specs = util.pickle_load_big('/Users/jose/PycharmProjects/the_speech/data/melspecs/melspec_dem_256')
warped_specs = []
for item in original_specs:
    warped_masked_spectrogram = \
        spec_augment_tensorflow.spec_augment(mel_spectrogram=item)
    warped_specs.append(warped_masked_spectrogram)

util.pickle_dump_big(warped_specs, '/Users/jose/PycharmProjects/the_speech/data/melspecs/melspec_dem_256_warped')
