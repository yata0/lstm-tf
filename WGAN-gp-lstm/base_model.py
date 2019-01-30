import tensorflow as tf
import numpy as np
import os
from hyperparams import TrainHyperParams as hp

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.saver = tf.train.Saver(var_list=tf.global_variables(scope="Graph"), max_to_keep=30)
        self.d_merge = tf.summary.merge([self.d_summary, self.d_real_summary, self.d_fake_summary, self.gp_summary])
        self.g_merge = tf.summary.merge([self.g_summary, self.g_ad_summary, self.g_mse_summary])
        self.d_vars = tf.global_variables(scope="Graph/discriminator")
        self.g_vars = tf.global_variables(scope="Graph/generator")
        self.embedding_vars = tf.global_variables(scope="Graph/embedding")
    def build_graph(self):
        with tf.variable_scope("Graph"):
            self.x = tf.placeholder(dtype=tf.int32,
                                    shape=[None, None],
                                    name="x_input")
            self.y = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, self.config.output_dims],
                                    name="real")
            self.length = tf.placeholder(dtype=tf.int32,
                                         shape=[None],
                                         name="length")

            self.x_inputs = self.get_embedding(self.x, self.config.emebedding_size, self.config.vocab_size)

            self.g_y =self.generator(self.x_inputs, self.length, "generator")
            # self.g_y_summary = tf.summary.histogram("generation", self.g_y)
            d_real = self.discriminator(self.x_inputs, self.y, self.length)
            d_fake = self.discriminator(self.x_inputs, self.g_y, self.length, reuse=True)
            self.d_real_loss = -tf.reduce_mean(d_real)
            self.d_fake_loss = tf.reduce_mean(d_fake)
            alpha = tf.random_uniform(shape=[tf.shape(self.x)[0],1,1], minval=0., maxval=1.)
            y_ = alpha * self.y + (1-alpha)*self.g_y
            grads = tf.gradients(self.discriminator(self.x_inputs, y_, self.length,reuse=True), [y_])[0]
            grads = tf.reshape(grads,[tf.shape[self.x][0],-1])
            grad_norm = tf.sqrt(tf.reduce_sum((grads)**2,axis=-1))
            self.gp_loss = 10 * tf.reduce_mean((grad_norm-1)**2)
            self.d_loss = self.d_real_loss + self.d_fake_loss + self.gp_loss
            self.g_mse_loss = self.get_L1_loss(self.y, self.g_y, self.length)
            self.g_ad_loss = -self.d_fake_loss
            # self.g_loss = self.g_mse_loss + self.g_ad_loss
            self.g_loss = self.g_ad_loss

            self.d_real_summary = tf.summary.scalar("d_real_loss", self.d_real_loss)
            self.d_fake_summary = tf.summary.scalar("d_fake_loss", self.d_fake_loss)
            self.gp_summary = tf.summary.scalar("gp", self.gp_loss)
            self.d_summary = tf.summary.scalar("d_loss", self.d_loss)
            self.g_mse_summary = tf.summary.scalar("g_mse_loss", self.g_mse_loss)
            self.g_ad_summary = tf.summary.scalar("g_ad_loss", self.g_ad_loss)
            self.g_summary = tf.summary.scalar("g_loss", self.g_loss)

    def generator(self, x, length, scope="generator"):
        with tf.variable_scope(scope):
            outputs = self.get_directional_output(x, length, self.config.cell_type, self.config.layer_size, self.config.units,"RNNModel")
            with tf.variable_scope("dropout"):
                outputs = tf.layers.dropout(outputs, rate=self.config.dropout_rate,
                                            training=tf.convert_to_tensor(self.config.training))
            outputs = tf.layers.dense(outputs, self.config.output_dims, activation=tf.nn.sigmoid, name="generator_output")
        return outputs

    def discriminator(self, x, y, length, reuse=None, scope="discriminator"):

        with tf.variable_scope(scope, reuse=reuse):

            with tf.variable_scope("InputLayer"):
                self.d_input = tf.concat(values=[x, y], axis=-1)
            outputs = self.get_directional_output(x, length, self.config.cell_type, self.config.layer_size, self.config.units,"RNNModel")
            with tf.variable_scope("dropout"):
                outputs = tf.layers.dropout(outputs, rate=self.config.dropout_rate,
                                            training=tf.convert_to_tensor(self.config.training))
            with tf.variable_scope("relevant_output"):
                batch_size = tf.shape(outputs)[0]
                max_length = tf.shape(outputs)[1]
                output_dims = outputs.get_shape().as_list()[-1]
                index = tf.range(0, batch_size) * max_length + length - 1
                flat = tf.reshape(outputs, [-1, output_dims])
                d_output = tf.gather(flat, index)

            with tf.variable_scope("discriminitor_output"):
                d_output = tf.layers.dense(inputs=outputs, units=1)

            return d_output
    
    def get_cell(self, cell_type, units, name):

        cell = None
        if cell_type == "lstm":
            cell = tf.contrib.rnn.BasicLSTMCell(units, name=name)
        elif cell_type == "gru":
            cell = tf.contrib.rnn.GRUCell(units, name=name)
        else:
            raise ValueError("the cell type:{} you want not in [{},{}]".format(cell_type, "lstm", "gru"))     
        return cell
    
    def get_multi_layer_cell(self,cell_type, layer_size, units, name):
        with tf.variable_scope(name):
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(
                [self.get_cell(cell_type, units, "{}_{}_{}".format(name,cell_type,i)) for i in range(layer_size)],
            )
        return multi_layer_cell

    def get_embedding(self, input, embedding_size, vocab_size):
        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.get_variable(
                name = "embedding_matrix",
                shape = [vocab_size, embedding_size],
                initializer=tf.random_normal_initializer(),
                dtype=tf.float32
            )
            output = tf.nn.embedding_lookup(self.embedding_matrix, input)
        return output


    def get_directional_output(self, input,length,cell_type, layer_size, units, name):
        with tf.variable_scope(name):
            multi_rnn_cell = self.get_multi_layer_cell(cell_type,layer_size,units,"MultiLayerCell")
            outputs, final_state = tf.nn.dynamic_rnn(
                cell = multi_rnn_cell,
                inputs = input,
                sequence_length = length,
                dtype = tf.float32
            )
        return outputs

 
    def get_L1_loss(self, label, result, length):
        loss = tf.reduce_mean(tf.abs(label-result), axis=-1)
        mask = tf.sequence_mask(length, self.config.max_len)
        mask = tf.to_float(mask)
        loss_ = tf.reduce_sum(loss * mask)/tf.reduce_sum(tf.to_float(length))
        return loss_
    def get_L2_loss(self, y, outputs, length):
        with tf.name_scope("loss_compute"):
            sequence_mask = tf.sequence_mask(length, self.config.max_len)
            sequence_mask = tf.to_float(sequence_mask)
            loss = tf.reduce_sum((y-outputs)**2, axis=-1)
            loss *= sequence_mask
            loss = tf.reduce_sum(loss)/tf.reduce_sum(sequence_mask)
        return loss

