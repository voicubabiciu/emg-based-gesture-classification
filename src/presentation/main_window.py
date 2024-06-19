# import pywt
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QComboBox, QSpinBox, \
    QFileDialog
from dependency_injector.wiring import inject

from src.domain.emg_payload import EMGPayload
from src.presentation.chart_plot import ChartPlot
from src.presentation.main_window_view_model import MainWindowViewModel


class MainWindow(QWidget):
    def __init__(self, view_model: MainWindowViewModel):
        super().__init__()
        self.view_model = view_model
        self.setWindowTitle("EMG Data processing")

        self.layout = QVBoxLayout()

        self.setLayout(self.layout)
        self.channel_one_graph = ChartPlot()
        self.channel_second_graph = ChartPlot()
        self.channel_third_graph = ChartPlot()
        self.channel_fourth_graph = ChartPlot()

        # first row for channel 1 and 2
        self.first_row = QHBoxLayout()

        self.chart_one_layout = QVBoxLayout()
        self.chart_one_layout.addWidget(QLabel('Channel 1'))
        self.chart_one_layout.addWidget(self.channel_one_graph)

        self.chart_second_layout = QVBoxLayout()
        self.chart_second_layout.addWidget(QLabel('Channel 2'))
        self.chart_second_layout.addWidget(self.channel_second_graph)

        self.first_row.addLayout(self.chart_one_layout)
        self.first_row.addLayout(self.chart_second_layout)

        # second row for channel 3 and 4
        self.second_row = QHBoxLayout()

        self.chart_third_layout = QVBoxLayout()
        self.chart_third_layout.addWidget(QLabel('Channel 3'))
        self.chart_third_layout.addWidget(self.channel_third_graph)

        self.chart_fourth_layout = QVBoxLayout()
        self.chart_fourth_layout.addWidget(QLabel('Channel 4'))
        self.chart_fourth_layout.addWidget(self.channel_fourth_graph)

        self.second_row.addLayout(self.chart_third_layout)
        self.second_row.addLayout(self.chart_fourth_layout)

        # training controls
        self.training_controls = QVBoxLayout()
        self.training_label = QLabel("Training")
        self.dataset_label = QComboBox(self)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(10, 1000)
        self.epochs_input.setValue(self.view_model.default_epochs)
        labels = ['Select a label', *self.view_model.available_labels]
        self.dataset_label.addItems(labels if labels is not None else [])
        self.previewDFButton = QPushButton('Load Dataset')
        self.previewDFButton.clicked.connect(self.open_file_picker)
        self.prepareDataButton = QPushButton('Train Keras model')
        self.prepareDataButton.clicked.connect(lambda: self.view_model.train_keras_model(int(self.epochs_input.text())))

        self.training_controls.addWidget(self.training_label)
        self.training_controls.addWidget(self.dataset_label)
        self.training_controls.addWidget(self.epochs_input)
        self.training_controls.addWidget(self.previewDFButton)
        self.training_controls.addWidget(self.prepareDataButton)

        # prediction controls
        self.prediction_controls = QVBoxLayout()
        self.prediction_label = QLabel("Prediction")
        self.combo_models = QComboBox(self)
        models = ['Select a trained model', *self.view_model.get_keras_models()]
        self.combo_models.addItems(models if models is not None else [])
        self.loadModelButton = QPushButton('Load Keras model')
        self.loadModelButton.clicked.connect(self.view_model.load_keras_model)

        self.prediction_controls.addWidget(self.prediction_label)
        self.prediction_controls.addWidget(self.combo_models)
        self.prediction_controls.addWidget(self.loadModelButton)
        self.openTensorboardButton = QPushButton('Open TensorBoard')
        self.openTensorboardButton.clicked.connect(self.open_tensorboard)
        self.prediction_controls.addWidget(self.openTensorboardButton)

        # adding training and prediction controls to a third row
        self.third_row = QHBoxLayout()
        self.third_row.addLayout(self.training_controls)
        self.third_row.addLayout(self.prediction_controls)

        self.layout.addLayout(self.first_row)
        self.layout.addLayout(self.second_row)
        self.layout.addLayout(self.third_row)
        self.statusLabel = QLabel()
        self.predictionLabel = QLabel()
        self.layout.addWidget(self.predictionLabel)
        self.layout.addWidget(self.statusLabel)

        self.view_model.status_stream.subscribe(lambda value: self.statusLabel.setText(value))
        self.view_model.prediction_stream.subscribe(lambda value: self.predictionLabel.setText(value))
        self.view_model.emg_data_stream.subscribe(self.update_plot_data)

        self.combo_models.currentIndexChanged.connect(
            lambda index: self.view_model.set_keras_model(self.sender().currentText() if index != 0 else None))
        self.dataset_label.currentIndexChanged.connect(
            lambda index: self.view_model.set_label(self.sender().currentText() if index != 0 else None))

    def open_file_picker(self):
        # Open file dialog and store the selected file path
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a dataset file",
            "",
            "All Files (*);;CSV Files (*.csv)"
        )

        # Update the label with the selected file path
        if file_path:
            self.statusLabel.setText(f"Selected dataset: {file_path}")
            self.view_model.dataset_path = file_path
        else:
            self.statusLabel.setText(f"Cannot load dataset")
            self.view_model.dataset_path = None

    def open_tensorboard(self):
        import webbrowser
        webbrowser.open(self.view_model.tensorboard_path)

    def update_plot_data(self, data: EMGPayload):
        self.channel_one_graph.update_plot_data(data.ch1)
        self.channel_second_graph.update_plot_data(data.ch2)
        self.channel_third_graph.update_plot_data(data.ch3)
        self.channel_fourth_graph.update_plot_data(data.ch4)
