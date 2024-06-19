import sys
import subprocess
import threading

from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from dependency_injector.wiring import inject, Provide

from src.core.containers import Container
from src.data.tensorboard_service import TensorboardService
from src.data.udp_client_service import UDPClientService
from src.presentation.main_window import MainWindow


def cleanup(tensorboard_service: TensorboardService):
    # Perform any necessary cleanup actions here
    print("Cleaning up before app closes.")
    tensorboard_service.stop()
    for thread in threading.enumerate():
        if isinstance(thread, threading.Thread):
            if isinstance(thread, UDPClientService):
                thread.stop()


@inject
def main(app: QtWidgets.QApplication = Provide[Container.qt_app],
         main_window: MainWindow = Provide[Container.main_window],
         tensorboard_service: TensorboardService = Provide[Container.tensorboard_service]) -> None:
    tensorboard_service.start()
    main_window.show()

    QCoreApplication.instance().aboutToQuit.connect(lambda _: cleanup(tensorboard_service))

    sys.exit(app.exec_())


def dispose_callback():
    print("Factory provider has been destroyed.")


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])
    main(*sys.argv[1:])
