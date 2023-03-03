import matplotlib.pyplot as plt


def plot_losses(train_loss, val_loss, output_path):
    """Plotting loss curves

    :param train_loss: training loss curve
    :param val_loss: validation loss curve (same length as training curve)
    :param output_path: optional path (finished with .png) where the plot is saved
    """
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()

    if output_path:
        plt.savefig(output_path)
        plt.clf()
    else:
        plt.show()
