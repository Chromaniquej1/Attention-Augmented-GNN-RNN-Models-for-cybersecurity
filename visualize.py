import matplotlib.pyplot as plt

def plot_attention_weights(attn_weights):
    plt.imshow(attn_weights.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Attention Weights")
    plt.show()

def plot_true_vs_predicted(y_true, y_pred):
    plt.scatter(range(len(y_true)), y_true, label="True Values", color='b', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted Values", color='r', alpha=0.6)
    plt.legend()
    plt.show()
