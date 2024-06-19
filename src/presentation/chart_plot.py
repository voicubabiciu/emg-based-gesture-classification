import pyqtgraph as pg


class ChartPlot(pg.PlotWidget):
    def __init__(self):
        pg.PlotWidget.__init__(self)
        self.setMouseEnabled(x=False, y=False)
        self.x = [0]
        self.y = [0]
        self.setFixedHeight(200)
        self.setYRange(1, 0, padding=0.05)
        pen = pg.mkPen(color=(0, 255, 0))
        self.data_line = self.plot(self.x, self.y, pen=pen, symbol='t', symbolPen='g', symbolSize=1)
        self.history_data = 0

    def update_plot_data(self, new_y):
        if len(self.x) > 1000:
            self.x = self.x[1:]  # Remove the first y element.

            self.y = self.y[1:]  # Remove the first
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.
        self.y.append(new_y)  # Add a new random value.

        if self.history_data >= 10:
            self.data_line.setData(self.x, self.y)
            self.history_data = 0
        else:
            self.history_data += 1

    def update_plot_data_x_y(self, new_x, new_y):
        self.x.append(new_x)  # Add a new value 1 higher than the last.
        self.y.append(new_y)  # Add a new random value.

        if self.history_data >= 10:
            self.data_line.setData(self.x, self.y)
            self.history_data = 0
        else:
            self.history_data += 1
