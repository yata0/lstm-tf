import numpy as np
import jieba
import os
import glob
import random
from matplotlib import pyplot as plt

# <PAD><0><UNKNOW>


class DataLoader:
    def __init__(self, char_path, files_path, sentence_path, data_dir, config):

        self.char2index, self.index2char = self.loadchar(char_path)
        self.sentences = self.load_sentence(sentence_path)
        self.file_list, _ = self.load_filelist(files_path)
        self.data_dir = data_dir
        self.blendshape = self.load_blendshape()
        self.hp = config

        self.padding_sentence, self.padding_blendshape, self.length = self.padding(self.hp.max_len)
        self.index_sentence = self.sentence2index()
        self.total_num = len(self.index_sentence)
        self.num_samples = self.total_num
        print("samples number: \t{}".format(self.total_num))

    def loadchar(self, datapath):
        char2index = {}
        index2char = {}
        with open(datapath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                char, index = line.split("\t")

                char2index[char] = int(index)
                index2char[int(index)] = char
        other = ["PAD", "UNKNOW"]
        for index, char in enumerate(other):
            print(char, index)
            char2index[char] = index
            index2char[index] = char
        return char2index, index2char

    def load_sentence(self, sentence_path):
        sentences = []
        with open(sentence_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line =line.strip()
                line = line.split(" ")
                sentences.append(line)
        return sentences

    def load_filelist(self, path):
        file_list = []
        lengths = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                file, length = line.split("\t")
                file_list.append(file)
                lengths.append(int(length))
            print("max length:{}".format(max(lengths)))
        return file_list, lengths

    def sentence2index(self):
        indexs = []
        for sentence in self.padding_sentence:
            index = []

            for word in sentence:
                index.append(self.char2index[word] if word in self.char2index else self.char2index["UNKNOW"])
            indexs.append(index)
        return indexs

    def load_blendshape(self):
        bs = []
        for file in self.file_list:
            file_name = os.path.basename(file)

            name, par_index, index, id, _ = file_name.split("_")

            blendshape_file = glob.glob(os.path.join(self.data_dir, "_".join([name, par_index, index]) + "_*blendshape.txt"))[0]
            
            blendshape = self.read_blendshape(blendshape_file)
            assert len(blendshape) != 0, "{}".format(blendshape_file)
            bs.append(blendshape)
        return bs

    def read_blendshape(self, filename):
        blendshape = np.loadtxt(filename)
        eyes = [1,3,5,9,13,15,16,18]
        blendshape = blendshape[:, eyes]
        blendshape = self.norm(np.asarray(blendshape, dtype=np.float32))
        return blendshape

    def norm(self, data):
        return data/100

    def padding(self, max_len):
        padding_sentence = []
        padding_blendshape = []
        length = []
        for sentence, blendshape in zip(self.sentences, self.blendshape):

            l = min([len(sentence), len(blendshape)])
            length.append(l)
            # print()
            while len(sentence) < max_len:
                sentence.append("PAD")
            N, D = blendshape.shape
            pad_BS = np.zeros([max_len, D])
            pad_BS[:l, :] = blendshape[:l, :]
            padding_sentence.append(sentence)
            padding_blendshape.append(pad_BS)
        return padding_sentence, padding_blendshape, length


    def get_test_batch(self, batch_size):
        data = list(zip(self.index_sentence, self.padding_blendshape, self.length, self.file_list))
        start_index = 0
        end_index = batch_size
        while start_index < self.total_num:
            end_index = min(self.total_num, end_index)
            batch = data[start_index:end_index]

            start_index += batch_size
            end_index += batch_size
            yield batch


    def get_batch(self, batch_size, phase="train"):
        data = list(zip(self.index_sentence, self.padding_blendshape, self.length))
        if phase == "train":
            random.shuffle(data)
        start_index = 0
        end_index = batch_size
        while end_index <= self.total_num:
            batch = data[start_index:end_index]
            temp = end_index
            start_index = temp
            end_index = end_index + batch_size
            yield batch


# if __name__ == "__main__":
#     test_data = DataLoader("../data/char2index.txt",
#                             "../data/train_list.txt",
#                             "../data/train_data.txt",
#                             "../data/spDB",hp)
#     print("length of blendshape:{}".format(len(test_data.blendshape)))
#     print("length of lines:{}".format(test_data.total_num))

