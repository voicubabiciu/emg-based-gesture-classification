"""Containers module."""
import sys

from dependency_injector import containers, providers
import influxdb_client
from PyQt5 import QtWidgets

from src.data.data_acquisition_repository import DataAcquisitionRepository
from src.data.database_service import DatabaseService
from src.data.tensorboard_service import TensorboardService
from src.data.udp_client_service import UDPClientService
from src.presentation.main_window import MainWindow
from src.presentation.main_window_view_model import MainWindowViewModel


class Container(containers.DeclarativeContainer):
    config = providers.Configuration(ini_files=["config.ini"])

    # logging = providers.Resource(
    #     logging.config.fileConfig,
    #     fname="logging.ini",
    # )

    qt_app = providers.Singleton(
        QtWidgets.QApplication,
        sys.argv
    )

    database_service = providers.Singleton(
        DatabaseService,
        url=config.database.url,
        token=config.database.api_token,
        org=config.database.org,
        bucket=config.database.bucket,
        use_database=config.database.use_database,
    )

    # Services
    udp_service = providers.Factory(
        UDPClientService,
        host=config.udp.host,
        port=config.udp.port
    )
    tensorboard_service = providers.Factory(
        TensorboardService,
        logdir=config.tensorboard.logdir,
        port=config.tensorboard.port
    )

    data_acquisition_repository = providers.Factory(
        DataAcquisitionRepository,
        udp_service=udp_service,
        database_service=database_service
    )

    main_window_view_model = providers.Factory(
        MainWindowViewModel,
        data_acquisition_repository=data_acquisition_repository,
        tensorboard_service=tensorboard_service,
        available_labels=config.cnn_features.available_labels,
        default_epochs=config.cnn_features.default_epochs,
    )

    main_window = providers.Factory(
        MainWindow,
        view_model=main_window_view_model,
    )
