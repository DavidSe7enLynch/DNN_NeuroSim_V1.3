import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = [10, 20, 30, 40, 50]
    y = [30, 30, 30, 30, 30]

    plt.plot(x, y, label="line 1")
    plt.legend()
    plt.show()

