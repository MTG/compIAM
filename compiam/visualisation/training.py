import matplotlib.pyplot as plt


def plot_losses(train_loss, val_loss, save_filepath):
    """TODO
    """
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.savefig(save_filepath)
    plt.clf()
    return
