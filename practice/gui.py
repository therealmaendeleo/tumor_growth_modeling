import os
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QComboBox,
    QGroupBox,
    QPushButton,
    QScrollArea,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatter
from model import tic_ode_system


def no_therapy(t):
    return 0.0


def constant_therapy(magnitude):
    return lambda t: magnitude


def pulsed_therapy(magnitude, start, duration):
    return lambda t: magnitude if start <= t <= start + duration else 0.0


def solve_rk4(ode, y0, t_span, h, args=()):
    t0, tf = t_span
    if h <= 0:
        raise ValueError("Шаг интегрирования должен быть > 0")
    n = int(np.ceil((tf - t0) / h))
    if n == 0:
        return np.array([t0]), np.array([y0])
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    yc = np.array(y0, dtype=float)
    for i in range(n):
        ti = t[i]
        k1 = h * ode(ti, yc, *args)
        k2 = h * ode(ti + h / 2, yc + k1 / 2, *args)
        k3 = h * ode(ti + h / 2, yc + k2 / 2, *args)
        k4 = h * ode(ti + h, yc + k3, *args)
        yc += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yc[yc < 0] = 0
        y[i + 1] = yc
    return t, y


class TicModelGUI(QWidget):
    MAX_SLIDER = 10000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование динамики роста опухоли при иммунотерапии")
        self.setFixedSize(1200, 700)
        self.is_init = True
        self.sliders = {}
        self.labels = {}
        self.param_metadata = {}
        self.param_configs = {}
        self.therapy_combos = {}
        self.therapy_magnitude_sliders = {}
        self.therapy_start_sliders = {}
        self.therapy_duration_sliders = {}
        self.therapy_widgets = {}
        self.figure_main = Figure()
        self.canvas_main = FigureCanvas(self.figure_main)
        self.figure_param = Figure()
        self.canvas_param = FigureCanvas(self.figure_param)
        self._init_ui()
        self._init_plots()
        self.run_simulation()
        self.is_init = False

    def _create_slider_row(
        self, short_txt, tooltip, vmin, vmax, vdef, step, slider_dict, label_dict, key
    ):
        row = QHBoxLayout()
        row.setSpacing(10)
        lbl = QLabel(short_txt)
        help_lbl = QLabel("?")
        help_lbl.setToolTip(tooltip)
        help_lbl.setStyleSheet(
            "QLabel { border: 1px solid grey; border-radius: 8px; "
            "background: #e8e8e8; padding: 1px; }"
            "QLabel:hover { background: #d0d0d0; }"
        )
        help_lbl.setFixedSize(16, 16)
        help_lbl.setAlignment(Qt.AlignCenter)
        label_box = QHBoxLayout()
        label_box.addWidget(lbl)
        label_box.addWidget(help_lbl)
        row.addLayout(label_box)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, self.MAX_SLIDER)
        if vmax > vmin:
            slider.setValue(int((vdef - vmin) / (vmax - vmin) * self.MAX_SLIDER))
        slider.setProperty("min_val", vmin)
        slider.setProperty("max_val", vmax)
        slider.setProperty("step", step)
        row.addWidget(slider)

        val_lbl = QLabel(f"{vdef:.4g}")
        val_lbl.setFixedWidth(60)
        row.addWidget(val_lbl)

        slider.valueChanged.connect(
            lambda v, s=slider, l=val_lbl: l.setText(f"{self._slider_to_value(s):.4g}")
        )
        slider.sliderReleased.connect(self.run_simulation)

        slider_dict[key] = slider
        label_dict[key] = val_lbl
        return row

    def _load_experimental_data(self):
        try:
            path = os.path.join("..", "data", "siu_1986_regression_low_dose.csv")
            data = np.genfromtxt(path, delimiter=",", skip_header=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            t_exp = data[:, 0]
            log_y_exp = data[:, 1]
            y_exp = 10**log_y_exp
            return t_exp, y_exp
        except Exception as e:
            print("Ошибка загрузки экспериментальных данных:", e)
            return None, None

    def _slider_to_value(self, slider):
        vmin = slider.property("min_val")
        vmax = slider.property("max_val")
        step = slider.property("step")
        if vmax == vmin:
            return vmin
        val = vmin + (slider.value() / self.MAX_SLIDER) * (vmax - vmin)
        if step > 0:
            val = round(val / step) * step
        return val

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.canvas_main)
        self.canvas_main.setFixedSize(650, 400)

        bottom_left = QWidget()
        bottom_layout = QHBoxLayout(bottom_left)

        params = {
            "a": (
                "Пролиферация (a):",
                "Скорость пролиферации опухолевых клеток (a), день⁻¹",
                0.01,
                2.0,
                0.18,
                0.01,
            ),
            "b": (
                "Обр. ёмкость среды (b):",
                "Обратная емкость среды (b), (кл.)⁻¹",
                1e-10,
                0.01,
                2e-9,
                1e-5,
            ),
            "c": (
                "Уничтожение T (c):",
                "Скорость уничтожения опухолевых клеток (c), мл/(кл·день)",
                2e-7,
                1e-5,
                1.101e-7,
                1e-7,
            ),
            "mu": (
                "Гибель I (μ):",
                "Скорость гибели иммунных клеток (μ), день⁻¹",
                0.01,
                0.5,
                0.0412,
                0.01,
            ),
            "d": (
                "Активация I (d):",
                "Скорость активации иммунных клеток (d), мл/(кл·день)",
                1e-9,
                1e-5,
                7.4e-14,
                1e-9,
            ),
            "p": (
                "Стим. цитокинами (p):",
                "Усиление пролиферации от цитокинов (p), мл/(кл·день)",
                0.0,
                1e-4,
                1e-5,
                1e-6,
            ),
            "lmbda": (
                "Распад цитокинов (λ):",
                "Скорость распада цитокинов (λ), день⁻¹",
                10.0,
                50.0,
                20.0,
                1.0,
            ),
        }
        initials = {
            "T0": ("Нач. T (T0):", "Опухолевые клетки (T0), кл/мл", 0, 1_000_000, 500_000.0, 1.0),
            "I0": ("Нач. I (I0):", "Иммунные клетки (I0), кл/мл", 0, 1_000_000, 320_000.0, 1.0),
            "C0": ("Нач. C (C0):", "Цитокины (C0), нг/мл", 0, 200, 0.0, 0.1),
        }
        times = {
            "t_start": ("Начало t:", "Начало (t_start), дни", 0.0, 10.0, 0.0, 0.1),
            "t_end": ("Конец t:", "Конец (t_end), дни", 10.0, 500.0, 120.0, 1.0),
            "h": ("Шаг h:", "Шаг интегрирования (h), дни", 0.01, 1.0, 0.1, 0.01),
        }

        for k, v in params.items():
            self.param_metadata[k] = v
        for k, v in initials.items():
            self.param_metadata[k] = v
        for k, v in times.items():
            self.param_metadata[k] = v

        g_params = QGroupBox("Параметры модели")
        g_params.setMaximumWidth(400)
        vb = QVBoxLayout(g_params)
        for k, (short, tip, mi, ma, de, st) in params.items():
            vb.addLayout(
                self._create_slider_row(short, tip, mi, ma, de, st, self.sliders, self.labels, k)
            )
        bottom_layout.addWidget(g_params)

        right_col = QWidget()
        right_col_layout = QVBoxLayout(right_col)
        g_init = QGroupBox("Начальные условия")
        vb = QVBoxLayout(g_init)
        for k, (short, tip, mi, ma, de, st) in initials.items():
            vb.addLayout(
                self._create_slider_row(short, tip, mi, ma, de, st, self.sliders, self.labels, k)
            )
        right_col_layout.addWidget(g_init)

        g_time = QGroupBox("Временной интервал и шаг")
        vb = QVBoxLayout(g_time)
        for k, (short, tip, mi, ma, de, st) in times.items():
            vb.addLayout(
                self._create_slider_row(short, tip, mi, ma, de, st, self.sliders, self.labels, k)
            )
        right_col_layout.addWidget(g_time)
        right_col_layout.addStretch(1)

        bottom_layout.addWidget(right_col)
        left_layout.addWidget(bottom_left)
        left_layout.addStretch(1)
        main_layout.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        therapy_types = ["Без терапии", "Постоянная", "Импульсная"]
        therapy_items = [
            ("eta_c", "Усиление цитотоксичности (ηc)", 0.0, 1e-4, 0.0, 1e-7, "мл/(клетка·день)"),
            ("eta_mu", "Снижение гибели иммунных клеток (ημ)", 0.0, 0.4, 0.0, 0.01, "день⁻¹"),
            ("s_A", "Адоптивный перенос (sA)", 0.0, 1e6, 0.0, 100, "кл/(мл·день)"),
            ("s_C", "Введение цитокинов (sC)", 0.0, 1e4, 0.0, 10, "нг/(мл·день)"),
        ]

        g_therapy = QGroupBox("Управление терапией")
        g_therapy.setFixedHeight(190)
        vb = QVBoxLayout(g_therapy)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for key, title, vmin, vmax, vdef, step, unit in therapy_items:
            box = QGroupBox(title)
            vb_box = QVBoxLayout(box)
            combo = QComboBox()
            combo.addItems(therapy_types)
            combo.currentIndexChanged.connect(lambda idx, k=key: self._update_therapy_ui(k, idx))
            vb_box.addWidget(combo)
            self.therapy_combos[key] = combo
            w_const = QWidget()
            vb_const = QVBoxLayout(w_const)
            vb_const.setContentsMargins(0, 5, 0, 0)
            vb_const.addLayout(
                self._create_slider_row(
                    "Величина:",
                    unit,
                    vmin,
                    vmax,
                    vdef,
                    step,
                    self.therapy_magnitude_sliders,
                    {},
                    f"{key}_const",
                )
            )
            vb_box.addWidget(w_const)
            self.therapy_widgets[f"{key}_const"] = w_const
            w_pulse = QWidget()
            vb_pulse = QVBoxLayout(w_pulse)
            vb_pulse.setContentsMargins(0, 5, 0, 0)
            vb_pulse.addLayout(
                self._create_slider_row(
                    "Величина:",
                    unit,
                    vmin,
                    vmax,
                    vdef,
                    step,
                    self.therapy_magnitude_sliders,
                    {},
                    f"{key}_pulse",
                )
            )
            vb_pulse.addLayout(
                self._create_slider_row(
                    "Начало:", "дни", 0, 100, 10, 1, self.therapy_start_sliders, {}, f"{key}_pulse"
                )
            )
            vb_pulse.addLayout(
                self._create_slider_row(
                    "Длит-ть:",
                    "дни",
                    0,
                    100,
                    5,
                    1,
                    self.therapy_duration_sliders,
                    {},
                    f"{key}_pulse",
                )
            )
            vb_box.addWidget(w_pulse)
            self.therapy_widgets[f"{key}_pulse"] = w_pulse
            scroll_layout.addWidget(box)
        scroll.setWidget(scroll_content)
        vb.addWidget(scroll)
        right_layout.addWidget(g_therapy)

        g_analysis = QGroupBox("Параметрический анализ")
        g_analysis.setFixedHeight(90)
        vb = QVBoxLayout(g_analysis)
        vb.setContentsMargins(10, 8, 10, 8)
        vb.setSpacing(6)
        hb = QHBoxLayout()
        hb.setSpacing(16)
        hb.addWidget(QLabel("Параметр X:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems(list(self.param_metadata.keys()))
        hb.addWidget(self.param_combo, stretch=1)
        hb.addSpacing(20)
        hb.addWidget(QLabel("Результат Y:"))
        self.output_combo = QComboBox()
        self.output_combo.addItems(["T (Опухолевые клетки)", "I (Иммунные клетки)", "C (Цитокины)"])
        hb.addWidget(self.output_combo, stretch=1)
        vb.addLayout(hb)
        btn = QPushButton("Запустить анализ")
        btn.clicked.connect(self.run_parametric_analysis)
        vb.addWidget(btn)
        right_layout.addWidget(g_analysis)
        right_layout.addWidget(self.canvas_param)
        self.canvas_param.setFixedSize(460, 380)
        main_layout.addWidget(right, 2)

        for key in self.therapy_combos:
            self._update_therapy_ui(key, self.therapy_combos[key].currentIndex())

    def _init_plots(self):
        ax = self.figure_main.add_subplot(111)
        ax.set_xlabel("Время, дни")
        ax.set_ylabel("Концентрация (кл/мл)")
        ax.grid(True)
        self.figure_main.tight_layout()
        axp = self.figure_param.add_subplot(111)
        axp.set_xlabel("Значение параметра")
        axp.set_ylabel("Конечное значение")
        axp.grid(True)
        self.figure_param.set_constrained_layout(True)

    def _update_therapy_ui(self, key, idx):
        self.therapy_widgets[f"{key}_const"].setVisible(idx == 1)
        self.therapy_widgets[f"{key}_pulse"].setVisible(idx == 2)
        if not self.is_init:
            self.run_simulation()

    def _get_therapy_func(self, key):
        idx = self.therapy_combos[key].currentIndex()
        if idx == 1:
            mag = self._slider_to_value(self.therapy_magnitude_sliders[f"{key}_const"])
            return constant_therapy(mag)
        if idx == 2:
            mag = self._slider_to_value(self.therapy_magnitude_sliders[f"{key}_pulse"])
            start = self._slider_to_value(self.therapy_start_sliders[f"{key}_pulse"])
            dur = self._slider_to_value(self.therapy_duration_sliders[f"{key}_pulse"])
            return pulsed_therapy(mag, start, dur)
        return no_therapy

    def run_simulation(self):
        try:
            p = {
                k: self._slider_to_value(s)
                for k, s in self.sliders.items()
                if k in {"a", "b", "c", "mu", "d", "p", "lmbda"}
            }
            y0 = [self._slider_to_value(self.sliders[k]) for k in ("T0", "I0", "C0")]
            t_start = self._slider_to_value(self.sliders["t_start"])
            t_end = self._slider_to_value(self.sliders["t_end"])
            h = self._slider_to_value(self.sliders["h"])

            if h <= 0 or t_end <= t_start:
                raise ValueError("Некорректный временной интервал или шаг")

            if h > (t_end - t_start):
                h = (t_end - t_start) / 100
                QMessageBox.information(self, "Внимание", f"Шаг уменьшен до {h:.4f}")

            args = (
                p["a"],
                p["b"],
                p["c"],
                p["mu"],
                p["d"],
                p["p"],
                p["lmbda"],
                self._get_therapy_func("eta_c"),
                self._get_therapy_func("eta_mu"),
                self._get_therapy_func("s_A"),
                self._get_therapy_func("s_C"),
            )

            t, y = solve_rk4(tic_ode_system, y0, (t_start, t_end), h, args)

            ax = self.figure_main.axes[0]
            ax.clear()
            ax.plot(t, y[:, 0], label="T (опухоль)", linestyle="-", linewidth=2)
            # ax.plot(t, y[:, 1], label="I (иммунитет)", linestyle="--", linewidth=2)

            t_exp, y_exp = self._load_experimental_data()
            if t_exp is not None:
                ax.scatter(t_exp, y_exp, marker="o", s=40, facecolors="none", edgecolors="red", label="Эксперимент (Siu 1986)")

                # --- НАЧАЛО БЛОКА РАСЧЕТА ОШИБОК ---

                # 1. Выбираем только те экспериментальные точки, которые попадают в наш временной интервал
                valid_indices = (t_exp >= t_start) & (t_exp <= t_end)
                t_exp_valid = t_exp[valid_indices]
                y_exp_valid = y_exp[valid_indices]

                if len(t_exp_valid) > 1: # Убедимся, что у нас есть точки для сравнения
                    # 2. Интерполируем значения модели на временные точки эксперимента
                    y_model_interp = np.interp(t_exp_valid, t, y[:, 0])

                    # 3. Рассчитываем RMSE
                    rmse = np.sqrt(np.mean((y_exp_valid - y_model_interp)**2))

                    # 4. Рассчитываем R^2
                    ss_res = np.sum((y_exp_valid - y_model_interp)**2)
                    ss_tot = np.sum((y_exp_valid - np.mean(y_exp_valid))**2)
                    if ss_tot > 0: # Избегаем деления на ноль, если все точки y_exp одинаковы
                        r_squared = 1 - (ss_res / ss_tot)
                    else:
                        r_squared = 1.0 # Если нет вариации, модель идеальна

                    # 5. Выводим текст с ошибками на график
                    error_text = f"RMSE: {rmse:,.0f}\n$R^2$: {r_squared:.3f}"
                    ax.text(0.95, 0.95, error_text,
                            transform=ax.transAxes,
                            fontsize=9,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.5))

                # --- КОНЕЦ БЛОКА РАСЧЕТА ОШИБОК ---

            ax.set_xlabel("Дни")
            ax.set_ylabel("Клетки (лог. шкала)")
            ax.set_yscale("symlog", linthresh=1e5)
            # ax.set_ylim(1e5, 1e9)
            ax.legend()
            ax.grid(True)
            self.canvas_main.draw()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def run_parametric_analysis(self):
        try:
            base_p = {
                k: self._slider_to_value(s)
                for k, s in self.sliders.items()
                if k in {"a", "b", "c", "mu", "d", "p", "lmbda"}
            }
            base_y0 = [self._slider_to_value(self.sliders[k]) for k in ("T0", "I0", "C0")]
            t_start = self._slider_to_value(self.sliders["t_start"])
            t_end = self._slider_to_value(self.sliders["t_end"])
            h = self._slider_to_value(self.sliders["h"])

            key_to_vary = self.param_combo.currentText()

            config_tuple = self.param_metadata.get(key_to_vary)
            if not config_tuple:
                QMessageBox.warning(
                    self, "Ошибка", f"Конфигурация для параметра {key_to_vary} не найдена"
                )
                return

            _, _, vmin, vmax, _, _ = config_tuple

            ax = self.figure_param.axes[0]
            ax.clear()

            if vmin > 0 and vmax / vmin > 100:
                values_to_test = np.logspace(np.log10(vmin), np.log10(vmax), 25)
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
            else:
                values_to_test = np.linspace(vmin, vmax, 25)
                ax.set_xscale("linear")

            results = []
            output_idx = self.output_combo.currentIndex()

            for val in values_to_test:
                current_p = base_p.copy()
                current_y0 = base_y0[:]

                if key_to_vary in current_p:
                    current_p[key_to_vary] = val
                elif key_to_vary in ("T0", "I0", "C0"):
                    current_y0[["T0", "I0", "C0"].index(key_to_vary)] = val

                args = (
                    current_p["a"],
                    current_p["b"],
                    current_p["c"],
                    current_p["mu"],
                    current_p["d"],
                    current_p["p"],
                    current_p["lmbda"],
                    self._get_therapy_func("eta_c"),
                    self._get_therapy_func("eta_mu"),
                    self._get_therapy_func("s_A"),
                    self._get_therapy_func("s_C"),
                )

                _, y = solve_rk4(tic_ode_system, current_y0, (t_start, t_end), h, args)

                results.append(y[-1, output_idx])

            ax.plot(values_to_test, results, "o-")
            ax.set_xlabel(f"Значение параметра: {key_to_vary}")
            ax.set_ylabel(f"Конечное значение: {self.output_combo.currentText()}")
            ax.grid(True)
            self.canvas_param.draw()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TicModelGUI()
    window.show()
    sys.exit(app.exec_())
