import subprocess


class TensorboardService:
    def __init__(self, logdir, port):
        self.logdir = logdir
        self.port = port
        self.tensorboard_process = None
        self.path = f"http://localhost:{self.port}"

    def start(self):
        if self.tensorboard_process is None:
            self.tensorboard_process = subprocess.Popen(
                ['tensorboard', '--logdir', self.logdir, '--port', self.port],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"TensorBoard started at http://localhost:{self.port}")

    def stop(self):
        if self.tensorboard_process:
            self.tensorboard_process.terminate()
            self.tensorboard_process = None
            print("TensorBoard stopped")
