import os
from config.opcode_vectorizer import OPCODE_Vectorizer
from config.model.Wide_Deep import WIDEVULDET
import pandas as pd
from config.model_metrics import LossHistory, Model_Metrics
from config.arg_parser import parameter_parser
import warnings
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")

args = parameter_parser()

class BaseClass:
    def __init__(self, args):
        self.args = args
        self.filename = "config/train_data/contracts.txt"
        # parse_file(filename)
        path = "./config/train_data/"
        self.base = os.path.splitext(os.path.basename(self.filename))[0]
        vector_filename = self.base + "_opcode_vectors.pkl"
        dataset = path + vector_filename

        if os.path.exists(dataset):
            self.df = pd.read_pickle(dataset)
        else:
            print('vectorizing opcodes . . .')
            self.df = self.vectorize_opcode(self.filename)
            self.df.to_pickle(dataset)

        if args.model == 'Wide_Deep':
            self.model = WIDEVULDET(self.df, name=self.args.model)

    def vectorize_opcode(self, filename):
        vector = []
        all_opcodes = []
        count = 0
        vectorizer = OPCODE_Vectorizer(self.args, filename)
        opcodes = vectorizer.byte2opcode()

        for opcode in opcodes:
            all_opcodes.append(opcode['opcode'])
            count += 1
            print("Collecting opcodes...", count, end="\r")
            vectorizer.add_to_fragment(opcode['opcode'])
            row = {"vector": opcode['opcode'], "val": opcode['val']}
            vector.append(row)
        # for testing and verification purposes only
        # vectorizer.write_to_test_file(args.test_file, fragments, "fragments and values")
        print("Training model...", end="\r")
        vectorizer.train_model()
        vectors = []
        count = 0
        for vec in vector:
            count += 1
            print("Processing opcodes...", count, end="\r")
            _vector = vectorizer.vectorize(vec["vector"])

            row = {"vector": _vector, "val": vec["val"]}
            vectors.append(row)

        df = pd.DataFrame(vectors)

        return df


def main():
    base = BaseClass(args)
    h = LossHistory()

    hist = base.model.train()
    pred, y_test = base.model.test()

    metrics = Model_Metrics(pred, y_test, hist)

    #print model metrics: accuracy, recall, precision, F1
    metrics.print_metrics()

    #plot receiver operating characteristics (ROC)
    metrics.plot_roc()

    #plot bar_chat comparing to other models
    metrics.plot_bar_chat()

    #plot confusion matrix
    metrics.plot_cm()



if __name__ == "__main__":
    main()
