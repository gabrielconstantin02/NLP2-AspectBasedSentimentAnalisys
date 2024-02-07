
class Config():
    def __init__(self):
        self.SEED = 8
        self.MAX_LENGTH = 128
        self.EPOCHS = 10
        self.NUM_LABELS = 4
        self.BATCH_SIZE = 25 # 1500 steps as 25 bs => 14 epochs on laptop
        # self.BATCH_SIZE = 16 # 1500 steps as 16 bs => 9 epochs on rest 
        self.NUM_LAYERS_FROZEN = 8
        self.MODEL_NAME = "laptop_bert_uncased_v4"
        self.LEARNING_RATE = 2e-5
        self.DATA_PATH = ''
        self.SAVE_PATH = './experiments'
        self.SAVE_FILES = ['models', 'utils', 'train.py', 'train.sh']
        self.SMALL_POSITIVE_CONST = 1e-4
        self.LABELS_MAPPING = {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}

CFG = Config()