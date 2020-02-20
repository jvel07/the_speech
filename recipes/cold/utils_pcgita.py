import os


# gets the 'subtask' folder name of the task
# e.g.: /home/egasj/PycharmProjects/the_speech/audio/sentences2/1_viste/non-normalized/hc/AVPEPUDEAC0001_viste.wav
# gets "1_viste", so that features can be saved separately from each 'subtask'
def get_real_parents(path_file):
    dir_of_file = os.path.dirname(path_file)
    parent_dir_of_file = os.path.dirname(dir_of_file)
    return os.path.dirname(parent_dir_of_file)