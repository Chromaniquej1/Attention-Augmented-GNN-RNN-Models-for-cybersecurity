import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GNN-RNN Model Training")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file')
    args = parser.parse_args()
    return args
