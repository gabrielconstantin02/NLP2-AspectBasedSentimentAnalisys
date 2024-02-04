
class Config():
    def __init__(self):
        self.SEED = 8
        self.MAX_LENGTH = 128
        self.EPOCHS = 10
        self.BATCH_SIZE = 16
        self.NUM_LAYERS_FROZEN = 8
        self.MODEL_NAME = "laptop_bert_uncased_v4"
        self.DATA_PATH = 'data/laptop14'

CFG = Config()