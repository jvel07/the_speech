import os


import numpy as np


# gets the 'subtask' folder name of the task
# e.g.: /home/egasj/PycharmProjects/the_speech/audio/sentences2/1_viste/non-normalized/hc/AVPEPUDEAC0001_viste.wav
# gets "1_viste", so that features can be saved separately from each 'subtask'
from common import util


def get_parent_level_2(path_file):
    dir_of_file = os.path.dirname(path_file)
    parent_dir_of_file = os.path.dirname(dir_of_file)
    return os.path.dirname(parent_dir_of_file)


# gets parent (directory) one more upper level
def get_parent_level_3(path_file):
    return os.path.dirname(get_parent_level_2(path_file))


# make labels according to wav directory: sentences2/1_viste/non-normalized/hc/AVPEPUDEAC0001_viste.wav
# where 'hc' is the label of the wav...
def make_labels(path_file):
    label = os.path.basename(os.path.dirname(path_file)).lower()
    wav = os.path.basename(os.path.splitext(path_file)[0])
    task = os.path.basename(get_parent_level_3(path_file))
    return wav, label, task


# save the labels. list of sets/tasks (NAME of the folders containing the audios), dir to the audios, output dir
def save_labels(list_sets, audio_dir, out_dir):
    for task in list_sets:
        list_of_wavs = util.traverse_dir(audio_dir + task, '.wav')
        labels_task = []
        for wav in list_of_wavs:
            w, label, task_name = make_labels(wav)
            labels_task.append(w + ' ' + label)
            # labels_task.sort()
        np.savetxt(out_dir + "labels_{}.txt".format(task), labels_task, delimiter=',', fmt='%s')