import os
class TrainHyperParams:
    def __init__(self):
        self.batch_size = 20
        self.cell_type = "lstm"
        self.dev_batch_size = 200
        self.vocab_size = 1593
        self.layer_size = 2
        self.units = 100
        self.embedding_size = 100
        self.layer_units = [64,32,16]
        self.dropout_rate = 0.5
        self.training = True
        self.epochs = 200
        self.max_len =543
        self.output_dims = 8
        self.reg = 0.1
        self.d_iter = 5
        self.learning_rate = 0.001
        self.data_dir = "../data/spDB"
        self.log_dir = "./WG-gp/eye/{}/L2/logs/2018_1120_de_l{}_b{}_h{}_lr{}".format(os.path.basename(self.data_dir),
                                                                                                    self.layer_size, self.batch_size, 
                                                                                                    self.units, self.learning_rate)
        self.save_dir = "./WG-gp/eye/{}/L2/ckpt/2018_1120_de_1{}_b{}_h{}_lr{}".format(os.path.basename(self.data_dir),self.layer_size, 
                                                                                                    self.batch_size, self.units, self.learning_rate)

class ContinueHyperParams(TrainHyperParams):
    def __init__(self):
        super(ContinueHyperParams, self).__init__()
        self.restore_dir = self.save_dir
        self.save_dir = self.restore_dir.replace("ckpt","continue_ckpt")
        self.learning_rate = 0.0001  
class TestHyperParams(TrainHyperParams):
    def __init__(self):
        super(TestHyperParams,self).__init__()
        self.training = False
        self.epochs = 1
        self.result_dir = self.log_dir.replace("logs", "result")

if __name__ == "__main__":
    test = TestHyperParams()
    print(test.result_dir)
    print(test.log_dir)
    continueTrain = ContinueHyperParams()
    print(continueTrain.save_dir)
    print(continueTrain.restore_dir)
    train = TrainHyperParams()
    print(train.data_dir)