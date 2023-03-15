import pandas as pd
import matplotlib.pyplot as plt


def plot_hw(hw_type, model_name, ylabel):
    file_name = f'hw/{hw_type}/{model_name}.csv'
    table = pd.read_csv(file_name)

    x = [3, 4, 5, 6, 7, 8]
    for key in table.keys():
        if key == 'ADCprec':
            continue
        y = table[key].tolist()
        plt.plot(x, y, label=key)

    # plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('ADC Resolution (bit)')

    plt.ylabel(ylabel)
    # plt.title(model_name)
    plt.savefig(f'hw/{hw_type}/{model_name}')
    plt.show()


if __name__ == '__main__':
    # ylabel = 'area (um^2)'
    # hw_type = 'area'
    # ylabel = 'latency (ns)'
    # hw_type = 'latency'
    # ylabel = 'energy（pJ) & power (uW)'
    # hw_type = 'power'
    ylabel = 'throughput & efficiency'
    hw_type = 'efficiency'
    plot_hw(hw_type, 'alexnet', ylabel)
    plot_hw(hw_type, 'resnet', ylabel)
    plot_hw(hw_type, 'densenet', ylabel)
