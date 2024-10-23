import argparse

def parse_args():
    """
    Parses command-line arguments for the GNN-RNN model training.

    Returns:
        argparse.Namespace: An object containing the parsed arguments, 
        specifically the 'data_path' argument which is the path to the dataset CSV file.
    """
    # Create an ArgumentParser object with a description of the script
    parser = argparse.ArgumentParser(description="GNN-RNN Model Training")

    # Add an argument 'data_path' that is required and specifies the path to the dataset
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file')

    # Parse the arguments and store them in an args variable
    args = parser.parse_args()

    # Return the parsed arguments
    return args

