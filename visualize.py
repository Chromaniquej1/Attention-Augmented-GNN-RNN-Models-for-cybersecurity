import matplotlib.pyplot as plt

def plot_attention_weights(attn_weights):
    """
    Plots the attention weights as a heatmap.

    Args:
        attn_weights (torch.Tensor): The attention weights to be visualized.
    """
    # Convert attention weights to numpy and plot as a heatmap
    plt.imshow(attn_weights.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add color bar to indicate weight intensity
    plt.title("Attention Weights")  # Set plot title
    plt.show()  # Display the plot

def plot_true_vs_predicted(y_true, y_pred):
    """
    Plots true values versus predicted values as scatter plots.

    Args:
        y_true (list or array-like): List of true labels/values.
        y_pred (list or array-like): List of predicted labels/values.
    """
    # Create a scatter plot of true values (in blue) and predicted values (in red)
    plt.scatter(range(len(y_true)), y_true, label="True Values", color='b', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted Values", color='r', alpha=0.6)
    plt.legend()  # Add a legend to differentiate between true and predicted values
    plt.title("True vs Predicted Values")  # Add title
    plt.show()  # Display the plot
