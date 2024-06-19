import pandas as pd
import time

from src.data.data_preprocesing_repository import preprocess_data
from src.data.database_service import DatabaseService

from src.data.neural_network_repository import train_model
from src.data.udp_client_service import UDPClientService
from src.domain.emg_payload import EMGPayload


class DataAcquisitionRepository:
    def __init__(self, udp_service: UDPClientService, database_service: DatabaseService):
        self.udp_service = udp_service
        self.database_service = database_service
        self.emg_data_stream = self.udp_service.data_stream

        self.dataFrame = pd.DataFrame([], columns=['label', 'time', 'ch1', 'ch2', 'ch3', 'ch4'])

    def add_data(self, data: EMGPayload, dataset_label: int):
        self.dataFrame.loc[len(self.dataFrame.index)] = [dataset_label, time.time(), data.ch1, data.ch2, data.ch3,
                                                         data.ch4]

    def train(self, epochs: int, dataset_path: str):
        if self.dataFrame.empty:
            self.dataFrame = pd.read_csv(dataset_path)
        else:
            from datetime import datetime
            self.dataFrame.to_csv(f'datasets/dataset-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', index=False)
        df, points = preprocess_data(self.dataFrame)
        self.database_service.store_dataset(points)
        train_model(df, epochs=epochs)

    def preview_dataframe(self):
        print(self.dataFrame)

    def load_dataframe(self, data_frame_path):
        self.dataFrame = pd.read_csv(data_frame_path)

    def get_keras_models(self):
        import os
        model_path = 'models'

        # Get list of directories
        dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]

        # Get the creation time for each directory
        dirs_with_times = [(d, os.path.getctime(os.path.join(model_path, d))) for d in dirs]

        # Sort the list of directories based on creation time
        sorted_dirs = sorted(dirs_with_times, key=lambda x: x[1], reverse=True)

        # Return only the directory names
        sorted_dirs_names = [d[0] for d in sorted_dirs]
        return sorted_dirs_names
