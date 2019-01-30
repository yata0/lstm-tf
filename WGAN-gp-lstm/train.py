import tensorflow as tf
from base_model import BaseModel
import numpy as np
import os
from dataloader import DataLoader
from hyperparams import TrainHyperParams 
import datetime

def mkdir(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

class Train(BaseModel):
    def __init__(self, train_data, dev_data, config):
        super(Train, self).__init__(config)
        self.train_data = train_data
        
        self.dev_data = dev_data
        self.merge = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.config.log_dir, "train"), tf.get_default_graph())
        self.dev_writer = tf.summary.FileWriter(os.path.join(self.config.log_dir, "test"))
        self.d_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.d_loss,
                                                                                var_list=self.d_vars)
        self.g_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.g_loss,
                                                                                var_list=self.g_vars.extend(self.embedding_vars))


    def train_epoch(self, epoch_index, sess):
        d_losses = 0
        g_losses = 0
        for index_iter, batch in enumerate(self.train_data.get_batch(self.config.batch_size)):
            unziped = list(zip(*batch))
            data, label, length = unziped
            
            input_feed = {self.x: data, self.y: label, self.length: length}
            d_output_feed = [self.d_loss, self.d_train_op, self.d_merge]
            g_output_feed = [self.g_loss, self.g_train_op, self.g_merge, self.g_y, self.g_mse_loss]
            for _iter in range(self.config.d_iter):
                d_loss, _, d_summary, _ ,l1_loss= sess.run(d_output_feed, feed_dict=input_feed)
                print("train interation:\t{}\tdiscriminator iter:\t{} \td_loss:\t{}".format(index_iter, _iter, d_loss))
            g_loss, _, g_summary, outputs,l1_loss = sess.run(g_output_feed,
                                                     feed_dict=input_feed)
            print("train interation:\t{} \tgenerator:\tloss:\t{}\tL1loss:{}".format(index_iter, g_loss,l1_loss))
            d_losses += d_loss
            g_losses += g_loss
            if index_iter % 10 == 0:
                self.train_writer.add_summary(d_summary, index_iter/10 + epoch_index * self.train_data.total_num/(self.config.batch_size *10))
                self.train_writer.add_summary(g_summary, index_iter/10 + epoch_index * self.train_data.total_num/(self.config.batch_size *10))
        print("Discriminator:train epoch {} loss:\t{}".format(epoch_index, d_losses / (index_iter+1)))
        print("Generator:train epoch {} loss:\t{}".format(epoch_index, g_losses / (index_iter + 1)))

    def test_step(self, sess):
        losses = 0
        results = []
        labels = []
        files = []

        for index_iter, batch in enumerate(self.train_data.get_test_batch(100)):
            unziped = list(zip(*batch))
            data, label, length, file_list = unziped
            files.extend(file_list)
            label = np.asarray(label)

            input_feed = {self.x: data, self.y: label, self.length: length}
            output_feed = [self.g_loss, self.g_y]
            loss, outputs = sess.run(output_feed, feed_dict =input_feed)
            label_, result = self.get_result(label, outputs, length)
            labels.extend(label_)
            results.extend(result)
            losses += loss
        print("test epoch  loss:\t{}".format(losses/(index_iter+1)))
        return files, labels, results

    def get_result(self, label, outputs, length):

        assert label.shape == outputs.shape, "label shape\t{},outputs shape\t{}".format(label.shape, outputs.shape)
        real_labels = []
        real_outputs = []
        for index in range(len(length)):
            label_ = label[index, :, :]
            output = outputs[index, :, :]
            length_ = length[index]
            real_label = label_[:length_,:]
            real_output = output[:length_,:]
            real_labels.append(real_label)
            real_outputs.append(real_output)
        return real_labels, real_outputs

if __name__ == "__main__":
    hp = TrainHyperParams()
    train_data = DataLoader("../data/char2index.txt",
                            "../data/train_list.txt",
                            "../data/train_data.txt",
                            hp.data_dir,hp)
    test_data = DataLoader("../data/char2index.txt",
                            "../data/test_list.txt",
                            "../data/test_data.txt",
                            hp.data_dir,hp)
    mkdir([hp.save_dir, os.path.join(hp.log_dir,"train"), os.path.join(hp.log_dir, "test")])
    train_model = Train(train_data, test_data, hp)

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        for epoch_index in range(hp.epochs):
            # files, labels, results = train_model.test_step(sess)
            # batch_save_file(files, labels, results, "./analyse2/epoch{}".format(epoch_index))
            train_model.train_epoch(epoch_index, sess)
            # train_model.dev_step()
            train_model.saver.save(sess, os.path.join(hp.save_dir, "{}_epoch_{}".format(datetime.date.today(), epoch_index)))