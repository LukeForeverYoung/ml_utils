def Histogram(data,bin=10,counted=False):
    from matplotlib import pyplot as plt
    plt.hist(data,bin=bin)
    plt.show()