import time

from rx.subject import BehaviorSubject

from src.data.data_acquisition_repository import DataAcquisitionRepository
from src.data.neural_network_repository import model_loaded, predict, load_trained_model
from src.data.tensorboard_service import TensorboardService
from src.domain.emg_payload import EMGPayload


class MainWindowViewModel:
    def __init__(self, data_acquisition_repository: DataAcquisitionRepository, tensorboard_service: TensorboardService,
                 available_labels, default_epochs):
        self.data_acquisition_repository = data_acquisition_repository
        self.dataset_label = None
        self.status_stream = BehaviorSubject('')
        self.prediction_stream = BehaviorSubject('')
        self.emg_data_stream = self.data_acquisition_repository.emg_data_stream
        self.emg_data_stream.subscribe(on_next=self.on_data_received)
        self.available_labels = available_labels.split(', ')
        self.timestamp = int(time.time() * 1000)
        self.selected_keras_model = None
        self.data_for_prediction = []
        self.dataset_path = None
        self.tensorboard_path = tensorboard_service.path
        self.default_epochs = int(default_epochs)

    def on_data_received(self, data: EMGPayload):
        now = int(time.time() * 1000)
        if model_loaded():
            if now - self.timestamp > 600:
                self.timestamp = now
                avg_data = EMGPayload.avg(self.data_for_prediction)
                predictions = predict(avg_data, EMGPayload.flatten(self.data_for_prediction))
                self.data_for_prediction.clear()
                if len(predictions) > 0:
                    self.status_stream.on_next(f'Prediction: {self.available_labels[predictions[0]]}')
            else:
                self.data_for_prediction.append(data)
        if self.dataset_label is not None:
            self.data_acquisition_repository.add_data(data, self.dataset_label)

    def set_label(self, new_val):
        if new_val:
            self.dataset_label = self.available_labels.index(new_val)
            self.status_stream.on_next(f'Set the label to: {new_val}({self.available_labels.index(new_val)})')
        else:
            self.status_stream.on_next(f'No label selected')

    def load_keras_model(self):
        if self.selected_keras_model is not None:
            load_trained_model(f'models/{self.selected_keras_model}/model.keras')
            # self.data_acquisition_repository.load_dataframe(f'models/{self.selected_keras_model}/data_frame.csv')
            self.status_stream.on_next('Keras model loaded')

    def train_keras_model(self, epochs: int, ):
        self.status_stream.on_next(f'Training in progress. Please wait!')
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._train_and_update_status(epochs))

    def _train_and_update_status(self, epochs: int):
        start_time = time.time()
        self.data_acquisition_repository.train(epochs, self.dataset_path)
        self.status_stream.on_next(f'Training finished. Took {round(time.time() - start_time, 2)}s!')

    def preview_dataframe(self):
        self.data_acquisition_repository.preview_dataframe()

    def get_keras_models(self):
        return self.data_acquisition_repository.get_keras_models()

    def set_keras_model(self, model):
        print(model)
        self.selected_keras_model = model
