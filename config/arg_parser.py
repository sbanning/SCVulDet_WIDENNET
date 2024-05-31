import argparse

path = "train_data/reent_contracts.txt"

def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='WIDENNET')

    parser.add_argument('-D', '--dataset', type=str, default=path,
                        choices=['train_data/contracts.txt'])
    parser.add_argument('-t', '--test_file', type=str, default='misc/out.txt')
    parser.add_argument('-M', '--model', type=str, default='Wide_Deep',
                        choices=['Wide_Deep', 'Other_Projects'])
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--vec_length', type=int, default=100, help='vector dimension')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size') #32
    parser.add_argument('-vt', '--vul_type', type=str, default='re_ent',
                        choices=['t_stamp', 're_ent'])

    return parser.parse_args()

