import argparse

path = "train_data/contracts.txt"

def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contract Vulnerability Detection Using Wide and Deep Neural Network')

    parser.add_argument('-D', '--dataset', type=str, default=path,
                        choices=['train_data/contracts.txt'])
    parser.add_argument('-t', '--test_file', type=str, default='misc/out.txt')
    parser.add_argument('-M', '--model', type=str, default='Wide_Deep',
                        choices=['Wide_Deep', 'Other_Projects'])
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--vec_length', type=int, default=300, help='vector dimension')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')

    return parser.parse_args()

