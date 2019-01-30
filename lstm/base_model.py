import tensorflow as tf
from dataloader import DataLoader
import numpy as np

class BaseModel:
    def __init__(self, config):
        self.hp = config
        self.regularizer = tf.contrib.layers.l2_regularizer(self.hp.reg)
        self.build_graph()
        self.reg_loss = tf.contrib.layers.apply_regularization(self.regularizer,tf.global_variables(scope="Graph"))
        
    def build_graph(self):

        self.x = tf.placeholder(dtype=tf.int32, name="x", shape=[None, None])
        self.y = tf.placeholder(dtype=tf.float32, name="y", shape=[None, None, self.hp.output_dims])
        self.length = tf.placeholder(dtype=tf.int32, name="length", shape=[None])
        self.x_input = self.get_embedding(self.x, self.hp.embedding_size, self.hp.vocab_size)

        with tf.variable_scope("Graph"):
            if self.hp.bidirectional:
                self.rnn_outputs = self.get_bidirectional_output(input = self.x_input,
                 length = self.length,
                cell_type = self.hp.cell_type, 
                layer_size = self.hp.layer_size, 
                units = self.hp.units, 
                name = "Bidirectional")
            else:
                self.rnn_outputs= self.get_directional_output(input = self.x_input,
                 length = self.length,
                cell_type = self.hp.cell_type, 
                layer_size = self.hp.layer_size, 
                units = self.hp.units, 
                name = "Directional")
            with tf.variable_scope("dropout"):
                self.outputs = tf.layers.dropout(inputs= self.rnn_outputs,
                rate = self.hp.dropout_rate,
                training = tf.convert_to_tensor(self.hp.training))

            with tf.variable_scope("final_output"):
                self.outputs = tf.layers.dense(inputs = self.outputs,
                 units = self.hp.output_dims,
                 activation = tf.nn.sigmoid)
            self.loss = self.get_loss(self.y, self.outputs, self.length)
            tf.summary.scalar("loss", self.loss)

            
    def get_cell(self, cell_type, units, name):
        """
        
        """
        cell = None
        if cell_type == "lstm":
            cell = tf.contrib.rnn.BasicLSTMCell(units, name=name)
        elif cell_type == "gru":
            cell = tf.contrib.rnn.BasicGRUCell(units, name=name)
        else:
            raise ValueError("the cell type:{} you want not in [{},{}]".format(cell_type, "lstm", "gru"))     
        return cell
    
    def get_multi_layer_cell(self,cell_type, layer_size, units, name):
        with tf.variable_scope(name):
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(
                [self.get_cell(cell_type, units, "{}_{}".format(cell_type,i)) for i in range(layer_size)]
            )
        return multi_layer_cell
    
    def get_bidirectional_cell(self,cell_type,layer_size,units, name):
        with tf.variable_scope(name):
            fw_cell = self.get_multi_layer_cell(cell_type, layer_size, units, "foward")
            bw_cell = self.get_multi_layer_cell(cell_type, layer_size, units, "backward")
        return fw_cell, bw_cell
    
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
    
    def get_bidirectional_output(self, input, length, cell_type, layer_size, units, name):
        with tf.variable_scope(name):
            fw_cell, bw_cell = self.get_bidirectional_cell(cell_type, layer_size, units, name)
            outputs, last_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = fw_cell,
                cell_bw = bw_cell,
                sequence_length = length,
                inputs = input,
                dtype = tf.float32
            )
            fw_output, bw_output = outputs
            fw_state, bw_state = last_state
            outputs = tf.concat([fw_output, bw_output], axis=-1)
        return outputs
    
    def get_embedding(self, input, embedding_size, vocab_size):
        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.get_variable(
                name = "embedding_matrix",
                shape = [vocal_size, embedding_size],
                initializer=tf.random_normal_initializer(),
                dtype=tf.float32
            )
            output = tf.nn.embedding_lookup(self.embedding_matrix, input)
        return output

    def get_loss(self, y, outputs, length):
        with tf.name_scope("loss_compute"):
            sequence_mask = tf.sequnece_mask(length, self.hp.max_len)
            sequence_mask = tf.to_float(sequence_mask)
            loss = tf.reduce_sum((y-outputs)**2, axis=-1)
            loss *= sequence_mask
            loss = tf.reduce_sum(loss)/tf.reduce_sum(sequence_mask)
        return loss


