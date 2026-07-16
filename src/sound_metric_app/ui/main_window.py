"""Minimal PySide6 desktop window: open a file, show the four metrics.

This is a functional starting point for the GUI phase, not the final design.
Run with:  python -m sound_metric_app.ui.main_window   (needs the 'gui' extra)
"""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from ..dsp import MetricsProcessor
from ..ingestion import read_frame


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sound Metric App")
        self.resize(560, 320)
        self._processor = MetricsProcessor()

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        open_btn = QtWidgets.QPushButton("Open DewesoftX file…")
        open_btn.clicked.connect(self._open_file)
        layout.addWidget(open_btn)

        self.file_label = QtWidgets.QLabel("No file loaded.")
        layout.addWidget(self.file_label)

        self.table = QtWidgets.QTableWidget(4, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(
            ["Peak dB", "Peak dBA", "Peak Impulse [prov.]", "LIAeq,100ms [prov.]"]
        ):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("—"))
        layout.addWidget(self.table)

        self.setCentralWidget(central)

    def _open_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open DewesoftX file", "", "Dewesoft files (*.dxd *.d7d)"
        )
        if not path:
            return
        try:
            frame = read_frame(path)
            result = self._processor.process(frame)
        except Exception as exc:  # noqa: BLE001 - surface any read/DSP error to the user
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return

        self.file_label.setText(f"{path}\n{result.channel} — {result.sample_rate:.0f} Hz")
        values = [
            result.peak_db,
            result.peak_dba,
            result.peak_impulse_db,
            result.liaeq_100ms_db,
        ]
        for row, val in enumerate(values):
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val:.2f} dB"))


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
