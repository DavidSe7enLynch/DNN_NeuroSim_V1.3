import pandas as pd
import matplotlib.pyplot as plt


class CsvPlotter:
    def __init__(self):
        self.file_dict = {
            'alexnet': 'imagenet_inference_adc_wlinput/alexnet-Table 1.csv',
            'resnet': 'imagenet_inference_adc_wlinput/resnet-54.02.csv',
            'densenet': 'imagenet_inference_adc_wlinput/densenet-56.15.csv'
        }
        self.file_name = None
        self.data = None

    def load_csv(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(file_name)


def plot_adc(data, model):
    x = [3, 4, 5, 6, 7, 8]
    for i in range(8):
        y = data[i].tolist()[1:]
        plt.plot(x, y, label=f"IN={i+1}")
    plt.legend(loc='upper left')
    plt.xlabel('ADC Resolution (bit)')
    plt.ylabel('Accuracy')
    plt.savefig(f'{model}_adc')
    plt.show()


def plot_wlinput(data, model):
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(3, 9):
        y = data[f"{i}"].tolist()
        plt.plot(x, y, label=f"ADC={i}")
    plt.legend(loc='upper left')
    plt.xlabel('Input Precision (bit)')
    plt.ylabel('Accuracy')
    plt.savefig(f'{model}_wlinput')
    plt.show()


def plot_acc(model):
    csv_plotter = CsvPlotter()
    csv_plotter.load_csv(csv_plotter.file_dict[model])
    data = csv_plotter.data
    plot_adc(data.transpose(), model)
    plot_wlinput(data, model)


if __name__ == '__main__':
    plot_acc('densenet')
    plot_acc('resnet')
    plot_acc('alexnet')

