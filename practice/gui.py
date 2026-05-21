import sys
from functools import partial

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,  # Или PySide6
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,  # Import QTabWidget
    QVBoxLayout,
    QWidget,
)
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from scipy.stats import linregress

from model import tic_ode_system

# 1. Переключаем основной шрифт на семейство с засечками (serif)
plt.rcParams["font.family"] = "serif"

# 2. Указываем использовать STIX General для обычного текста с засечками
plt.rcParams["font.serif"] = ["STIXGeneral", "Times New Roman", "DejaVu Serif"]

# 3. САМОЕ ВАЖНОЕ: переключаем математический шрифт (для формул в $...$) на STIX
plt.rcParams["mathtext.fontset"] = "stix"

plt.rcParams.update({"font.size": 20})
plt.rcParams["axes.labelsize"] = 26


def calculate_oscillation_period(t, T):
    # Находим индексы пиков концентрации опухолевых клеток
    # prominence помогает отсечь мелкий численный шум
    peaks, _ = find_peaks(T, prominence=np.max(T) * 0.1)

    if len(peaks) < 2:
        return 0  # Колебаний нет или всего один пик

    # Вычисляем разницу во времени между последовательными пиками
    peak_times = t[peaks]
    periods = np.diff(peak_times)

    return np.mean(periods)  # Средний период в днях


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.fit_call_count = 0
        self.setWindowTitle("Моделирование динамики опухолевых процессов")
        self.load_clinical_data()

        self.param_list = [
            (
                "a",
                "a (рост)",
                "[день⁻¹]",
                "Скорость пролиферации опухолевых клеток",
                0.01,
                1.0,
                0.18,
                0.01,
            ),
            (
                "b",
                "b (ёмкость)",
                "[(кл.)⁻¹]",
                "Параметр, обратный емкости среды",
                1e-10,
                1e-8,
                2e-9,
                1e-10,
            ),
            (
                "c",
                "c (киллинг)",
                "[мл/(кл·день)]",
                "Константа скорости уничтожения опухолевых клеток",
                1e-10,
                1e-5,
                1.1e-7,
                1e-10,
            ),
            (
                "mu",
                "μ (гибель I)",
                "[день⁻¹]",
                "Константа скорости гибели иммунных клеток",
                0.1,
                0.5,
                0.0412,
                0.01,
            ),
            (
                "d",
                "d (активация)",
                "[мл/(кл·день)]",
                "Константа скорости активации иммунных клеток",
                1e-10,
                1e-5,
                1.1e-7,
                1e-10,
            ),
            (
                "p",
                "p (стимуляция)",
                "[мл/(кл·день)]",
                "Константа усиления пролиферации под действием цитокинов",
                1e-12,
                1e-7,
                4e-8,
                1e-12,
            ),
            (
                "lmbda",
                "λ (распад цит.)",
                "[день⁻¹]",
                "Константа скорости распада цитокинов",
                10.0,
                50.0,
                20.0,
                1.0,
            ),
            (
                "eta_c_val",
                "Усиление цитотокс. (ηc)",
                "[мл/(кл·день)]",
                "Усиление киллинга (до 10*c)",
                0,
                1e-5,
                1.1e-7,
                1e-10,
            ),
            (
                "eta_mu_val",
                "Подавление гибели (ημ)",
                "[день⁻¹]",
                "Снижение смертности I (не более 0.8*μ)",
                0,
                0.4,
                0.01,
                0.001,
            ),
            (
                "sC_val",
                "Скорость ввода цит. (sC)",
                "[нг/(мл·день)]",
                "Скорость введения цитокинов",
                0,
                1e4,
                100,
                10,
            ),
            (
                "sA_val",
                "Скорость ввода кл. (sA)",
                "[кл/(мл·день)]",
                "Скорость введения иммунных клеток",
                0,
                1e5,
                1e2,
                10,
            ),
        ]
        self.init_list = [
            (
                "T0",
                "T0",
                "[кл/мл]",
                "Начальная концентрация опухолевых клеток",
                1e4,
                1e8,
                5e5,
                1e4,
            ),
            (
                "I0",
                "I0",
                "[кл/мл]",
                "Начальная концентрация иммунных эффекторных клеток",
                0,
                1e6,
                3.2e5,
                1000,
            ),
            (
                "C0",
                "C0",
                "[нг/мл]",
                "Начальная концентрация цитокинов",
                0,
                200,
                0.1,
                0.1,
            ),
            (
                "t_end",
                "t_end",
                "[Дни]",
                "Конечное время моделирования",
                10,
                900,
                120,
                10,
            ),
            # Внутри self.init_list в MainWindow.__init__
            (
                "sc_start",
                "Начало курса (sC)",
                "[день]",
                "День первого введения цитокинов",
                0,
                100,
                10,  # Начальное значение
                1,
            ),
            (
                "sc_duration",
                "Срок курса (sC)",
                "[дней]",
                "Продолжительность ежедневной терапии",
                1,
                60,
                14,  # Начальное значение
                1,
            ),
        ]

        self.sliders = {}
        self.value_labels = {}
        self.colorbar_ax = None
        self.show_clinical_checked = True  # Состояние по умолчанию

        # --- Main Tab Widget ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Tab 1: Simulation ---
        self.simulation_tab = QWidget()
        self.setup_simulation_tab()
        self.tabs.addTab(self.simulation_tab, "Визуализация модели")

        # --- Tab 2: Parametric Analysis ---
        self.analysis_tab = QWidget()
        self.setup_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "Параметрический анализ")

        self.update_plot()
        self.showMaximized()

    def toggle_oy_visibility(self):
        """Переключает видимость полей для 2D анализа"""
        is_heatmap = self.analysis_mode_combo.currentText() == "Тепловая карта (2D)"
        self.oy_settings_widget.setVisible(is_heatmap)
        self.analysis_param_combo2.setVisible(is_heatmap)
        # Находим label для combo2 через форму, если нужно:
        label = self.analysis_param_combo2.parentWidget().findChild(QLabel, "")  # упрощенно

    def setup_simulation_tab(self):
        main_layout = QHBoxLayout(self.simulation_tab)

        # Левая часть: График
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        self.sim_canvas = MplCanvas(self)
        plot_layout.addWidget(self.sim_canvas)

        # Правая часть: Контроллеры
        controls_frame = QFrame()
        controls_frame.setFixedWidth(450)
        controls_outer_layout = QVBoxLayout(controls_frame)
        controls_outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content_widget = QWidget()
        controls_main_layout = QVBoxLayout(scroll_content_widget)

        # 1. Группа параметров
        g_params = QGroupBox("Параметры модели")
        params_vbox = QVBoxLayout(g_params)
        for params in self.param_list:
            params_vbox.addWidget(self._create_slider_row(*params))
        controls_main_layout.addWidget(g_params)

        # 2. Группа начальных условий
        g_init = QGroupBox("Начальные условия и время")
        init_vbox = QVBoxLayout(g_init)
        for params in self.init_list:
            init_vbox.addWidget(self._create_slider_row(*params))
        controls_main_layout.addWidget(g_init)

        # 3. Группа ТЕРАПИИ (Вот она)
        g_therapy = QGroupBox("Режим воздействия")
        therapy_layout = QVBoxLayout(g_therapy)

        # Сам выбор терапии
        self.therapy_combo = QComboBox()
        self.therapy_combo.addItems(
            [
                "Без терапии",
                "Иммунотерапия (sC)",
                "Адоптивная (sA)",
                "Ингибирование (ηc, ημ)",
                "Комбинированная",
            ]
        )
        self.therapy_combo.currentIndexChanged.connect(self.update_plot)
        therapy_layout.addWidget(self.therapy_combo)

        # Настройки тайминга
        timing_layout = QFormLayout()

        self.start_day_input = QSpinBox()
        self.start_day_input.setRange(0, 1000)
        self.start_day_input.setValue(20)
        self.start_day_input.setSuffix(" день")
        self.start_day_input.valueChanged.connect(self.update_plot)

        self.interval_input = QSpinBox()
        self.interval_input.setRange(1, 200)
        self.interval_input.setValue(20)
        self.interval_input.setSuffix(" дн.")
        self.interval_input.valueChanged.connect(self.update_plot)

        self.count_input = QSpinBox()
        self.count_input.setRange(1, 50)
        self.count_input.setValue(3)
        self.count_input.setSuffix(" раз")
        self.count_input.valueChanged.connect(self.update_plot)

        timing_layout.addRow("Начало терапии:", self.start_day_input)
        timing_layout.addRow("Интервал между:", self.interval_input)
        timing_layout.addRow("Количество доз:", self.count_input)

        therapy_layout.addLayout(timing_layout)

        # Добавляем готовую группу в основной макет один раз
        controls_main_layout.addWidget(g_therapy)

        # 4. Группа AUC
        g_auc = QGroupBox("Анализ токсичности (AUC)")
        auc_layout = QFormLayout(g_auc)
        self.auc_day_input = QSpinBox()
        self.auc_day_input.setRange(1, 1000)
        self.auc_day_input.setValue(120)
        self.auc_day_input.valueChanged.connect(self.update_plot)
        self.auc_result_label = QLabel("AUC: ---")
        self.auc_result_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        auc_layout.addRow("Расчет до дня:", self.auc_day_input)
        auc_layout.addRow(self.auc_result_label)
        controls_main_layout.addWidget(g_auc)

        # 5. Кнопки и чекбоксы
        self.show_clinical_cb = QCheckBox("Показывать клинические данные")
        self.show_clinical_cb.setChecked(True)
        self.show_clinical_cb.stateChanged.connect(self.update_plot)
        controls_main_layout.addWidget(self.show_clinical_cb)

        fit_button = QPushButton("Подобрать параметры (c, d)")
        fit_button.clicked.connect(self._fit_parameters)
        controls_main_layout.addWidget(fit_button)

        controls_main_layout.addStretch()

        # 1. Создаем кнопку
        self.btn_save_svg = QPushButton("Сохранить график в SVG")

        # 2. Если хотите, можно задать ей тот же 14-й шрифт для единообразия
        font = self.btn_save_svg.font()
        font.setPointSize(12)  # Кнопку можно чуть меньше, чем график
        self.btn_save_svg.setFont(font)

        # 3. Добавляем кнопку в ваш layout (слой) под графиком sim_canvas
        # Например, если у вас там QVBoxLayout:
        # validation_layout.addWidget(self.sim_canvas)
        controls_main_layout.addWidget(self.btn_save_svg)

        # 4. Привязываем клик к методу сохранения
        self.btn_save_svg.clicked.connect(self.save_plot_to_svg)

        scroll_area.setWidget(scroll_content_widget)
        controls_outer_layout.addWidget(scroll_area)

        main_layout.addWidget(plot_frame, 1)
        main_layout.addWidget(controls_frame)

    def save_plot_to_svg(self):
        # 1. Открываем диалог сохранения файла с фильтром только на .svg
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график",
            "model_validation_plot.svg",  # Имя файла по умолчанию
            "Векторная графика (*.svg)",  # Фильтр расширений
        )

        # 2. Если пользователь не нажал "Отмена" и выбрал путь
        if file_path:
            try:
                # Магия matplotlib: сохраняем фигуру из холста
                # bbox_inches='tight' гарантирует, что крупные шрифты STIX не обрежутся по краям
                self.sim_canvas.fig.savefig(file_path, format="svg", bbox_inches="tight")

                # Показываем красивое уведомление об успешном сохранении
                QMessageBox.information(self, "Успех", f"График успешно сохранен в:\n{file_path}")

            except Exception as e:
                # На случай, если файл открыт в другой программе или нет прав на запись
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def save_analysis_plot(self):
        """Сохранение текущего графика в векторный формат SVG"""
        # Формируем предлагаемое имя файла на основе выбранной метрики
        metric_name = self.analysis_metric_combo.currentText().replace(" ", "_")

        # Открываем диалоговое окно для выбора пути
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график",
            f"analysis_{metric_name}.svg",
            "SVG Files (*.svg);;PNG Files (*.png);;All Files (*)",
        )

        if file_path:
            try:
                # Сохраняем всю фигуру целиком
                # bbox_inches='tight' убирает лишние белые поля вокруг графика
                self.analysis_canvas.fig.savefig(file_path, format="svg", bbox_inches="tight")
                print(f"График успешно сохранен: {file_path}")
            except Exception as e:
                print(f"Ошибка при сохранении: {e}")

    def setup_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)

        # Создаем основной вертикальный лейаут для настроек
        self.main_settings_layout = QVBoxLayout()

        controls_group = QGroupBox("Настройки анализа")
        controls_layout = QHBoxLayout(controls_group)
        form_layout = QFormLayout()

        # 1. Инициализация комбобоксов
        self.analysis_param_combo = QComboBox()
        self.analysis_param_combo2 = QComboBox()
        self.analysis_mode_combo = QComboBox()
        self.analysis_metric_combo = QComboBox()

        # 2. Наполнение данными
        all_param_names = [p[1] for p in self.param_list] + [p[1] for p in self.init_list]
        self.analysis_param_combo.addItems(all_param_names)
        self.analysis_param_combo2.addItems(all_param_names)
        self.analysis_param_combo2.setCurrentIndex(2)

        self.analysis_mode_combo.addItems(["Семейство кривых (1D)", "Тепловая карта (2D)"])

        self.metrics_map = {
            "Пиковая концентрация": "max",
            "Время до пика (дни)": "t_max",
            "Скорость роста (Log10/день)": "growth",
            r"$AUC_{norm}$": "auc",
            "Период рецидивов (дни)": "period",  # ДОБАВЛЕНО
            "Амплитуда \nколебаний": "amplitude",  # ДОБАВЛЕНО
        }
        self.analysis_metric_combo.addItems(self.metrics_map.keys())

        # 3. Настройка числовых вводов
        self.analysis_range_start = QDoubleSpinBox()
        self.analysis_range_start.setRange(0.001, 100.0)
        self.analysis_range_start.setValue(0.5)

        self.analysis_range_end = QDoubleSpinBox()
        self.analysis_range_end.setRange(0.01, 1000.0)
        self.analysis_range_end.setValue(2.0)

        self.analysis_steps = QSpinBox()
        self.analysis_steps.setRange(3, 50)
        self.analysis_steps.setValue(10)

        # 4. Секция OY (только для Heatmap)
        self.oy_settings_widget = QWidget()  # Создаем контейнер для удобного скрытия
        oy_layout = QHBoxLayout(self.oy_settings_widget)
        self.label_oy = QLabel("Диапазон OY (множитель):")
        self.oy_min = QDoubleSpinBox()
        self.oy_max = QDoubleSpinBox()
        self.oy_min.setValue(0.5)
        self.oy_max.setValue(2.0)
        self.oy_max.setMaximum(1000000.0)
        oy_layout.addWidget(self.label_oy)
        oy_layout.addWidget(self.oy_min)
        oy_layout.addWidget(self.oy_max)

        # 5. Добавление в форму
        form_layout.addRow("Режим анализа:", self.analysis_mode_combo)
        form_layout.addRow("Метрика:", self.analysis_metric_combo)
        form_layout.addRow("Параметр X:", self.analysis_param_combo)

        range_layout = QHBoxLayout()
        range_layout.addWidget(self.analysis_range_start)
        range_layout.addWidget(QLabel("до"))
        range_layout.addWidget(self.analysis_range_end)
        form_layout.addRow("Диапазон X (отн.):", range_layout)

        form_layout.addRow("Параметр Y (2D):", self.analysis_param_combo2)
        form_layout.addRow(self.oy_settings_widget)
        form_layout.addRow("Шагов расчета:", self.analysis_steps)

        # 6. Кнопка и сигналы
        self.analysis_run_button = QPushButton("Запустить расчет")
        self.analysis_run_button.clicked.connect(self._run_analysis)

        self.save_button = QPushButton("Сохранить SVG")
        self.save_button.clicked.connect(self.save_analysis_plot)

        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.analysis_run_button)
        controls_layout.addWidget(self.save_button)  # Добавьте эту строку

        layout.addWidget(controls_group)

        # Холст
        self.analysis_canvas = MplCanvas(self)
        layout.addWidget(self.analysis_canvas)

        # ПОДКЛЮЧЕНИЕ СИГНАЛА ВИДИМОСТИ
        self.analysis_mode_combo.currentIndexChanged.connect(self.toggle_oy_visibility)
        self.toggle_oy_visibility()  # Вызвать сразу для настройки начального вида

    def _calculate_metric(self, t, T, metric_type):
        """Вычисляет выбранную статистическую метрику по кривой роста опухоли"""
        if len(T) == 0:
            return 0

        if metric_type == "max":
            return np.max(T)

        elif metric_type == "t_max":
            return t[np.argmax(T)]

        elif metric_type == "growth":
            # Берем первые 10% времени для оценки начальной скорости
            idx = max(2, len(t) // 10)
            t_start, T_start = t[:idx], T[:idx]

            # Наклон в логарифмической шкале (Log10 единиц в день)
            valid = T_start > 0
            if np.sum(valid) > 1:
                # linregress возвращает (slope, intercept, rvalue, pvalue, stderr)
                slope, _, _, _, _ = linregress(t_start[valid], np.log10(T_start[valid]))
                return slope
            return 0

        elif metric_type == "auc":
            # Проверка: фильтруем неположительные значения для логарифмической AUC
            mask = T > 0
            if np.sum(mask) < 2:
                return 0
            # Убираем np.log10, чтобы метрика соответствовала AUC_LIMIT
            return np.trapezoid(T[mask], t[mask])

        elif metric_type == "period":
            return calculate_oscillation_period(t, T)

        elif metric_type == "amplitude":  # НОВАЯ МЕТРИКА
            # Амплитуда как разность между пиком и глубокой ремиссией
            # В логарифмических моделях иногда полезнее считать лог-амплитуду,
            # но для диплома лучше оставить линейную разность:
            t_max = np.max(T)
            t_min = np.min(T)
            return t_max - t_min

        return 0

    def load_clinical_data(self):
        try:
            df = pd.read_csv("data/siu_low.csv")
            df = df.rename(columns={"log10_cells": "linear_cells"})
            Y, E = df["linear_cells"].values, df["y_error"].values
            valid_indices = (Y > 0) & (Y - E >= 0)
            df = df.loc[valid_indices].copy()
            Y, E = df["linear_cells"].values, df["y_error"].values
            D_log = np.log10(1 + E / Y)
            Y_lower_linear_bound = Y / (10**D_log)
            Y_upper_linear_bound = Y * (10**D_log)
            self.clinical_data_yerr = np.array([Y - Y_lower_linear_bound, Y_upper_linear_bound - Y])
            self.clinical_data = df
        except Exception as e:
            self.clinical_data = None
            print(f"Error loading clinical data: {e}")

    def run_simulation(self, params, y0, t_span):
        a, b, c, mu, d, p, lmbda = params[:7]
        v1, v2, v3, v4 = self.get_therapy_functions()

        def ode(t, y):
            # Передаем всё по порядку
            return tic_ode_system(t, y, a, b, c, mu, d, p, lmbda, v1, v2, v3, v4)

        # Создаем сетку времени от 0 до конца t_span
        t_eval_points = np.linspace(t_span[0], t_span[1], 500)

        sol = solve_ivp(
            ode,
            t_span,
            y0,
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            t_eval=t_eval_points,
            vectorized=True,  # Если твоя функция системы поддерживает это
        )
        return (
            (sol.t, sol.y.T)
            if sol.success
            else (np.array([0.0, t_span[1]]), np.full((len(t_span), 3), np.nan))
        )

    def _create_slider_row(self, key, name, units, tooltip, min_val, max_val, init_val, step):
        row_widget = QWidget()
        row_layout = QVBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 5, 0, 5)
        label_layout = QHBoxLayout()
        name_label = QLabel(f"<b>{name}</b> {units}")
        tooltip_label = QLabel("(?)")
        tooltip_label.setToolTip(tooltip)
        label_layout.addWidget(name_label)
        label_layout.addStretch()
        label_layout.addWidget(tooltip_label)
        slider_layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        multiplier = 1.0 / step if step > 0 else 1.0
        slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
        slider.setValue(int(init_val * multiplier))
        slider.setSingleStep(1)
        val_text = self._format_value(key, init_val)
        value_label = QLabel(val_text)
        value_label.setFixedWidth(80)
        value_label.setAlignment(Qt.AlignRight)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        row_layout.addLayout(label_layout)
        row_layout.addLayout(slider_layout)
        slider.valueChanged.connect(partial(self.update_label_text, key))
        slider.sliderReleased.connect(self.update_plot)
        self.sliders[key] = (slider, multiplier)
        self.value_labels[key] = value_label
        return row_widget

    def _format_value(self, key, value):
        if key in ["T0", "I0", "t_end"]:
            return f"{value:,.0f}"
        if value < 0.01 and value != 0:
            return f"{value:.1e}"
        return f"{value:,.2f}"

    def get_therapy_functions(self):
        mode = self.therapy_combo.currentText()

        # Считываем интенсивность из слайдеров
        sC_amp = self.get_slider_value("sC_val")
        sA_amp = self.get_slider_value("sA_val")
        ec_amp = self.get_slider_value("eta_c_val")
        em_amp = self.get_slider_value("eta_mu_val")

        # Считываем параметры ГРАФИКА ТЕРАПИИ из SpinBox'ов интерфейса
        t_start = self.start_day_input.value()
        t_interval = self.interval_input.value()
        t_count = self.count_input.value()

        # Параметры для sC (цитокины) из слайдеров
        sc_start_slider = self.get_slider_value("sc_start")
        sc_duration_slider = self.get_slider_value("sc_duration")
        sc_end_slider = sc_start_slider + sc_duration_slider

        def zero(t):
            return 0.0

        self.active_injection_days = []
        v1, v2, v3, v4 = zero, zero, zero, zero

        if mode == "Без терапии":
            return v1, v2, v3, v4

        # 1. Монотерапия цитокинами (sC)
        elif mode == "Иммунотерапия (sC)":
            self.active_injection_days = [
                d for d in range(int(sc_start_slider), int(sc_end_slider))
            ]
            v3 = lambda t: (
                sC_amp if (sc_start_slider <= t <= sc_end_slider and t % 1 < 0.5) else 0.0
            )

        # 2. Адоптивная терапия (sA) - ТЕПЕРЬ ДИНАМИЧЕСКАЯ
        elif mode == "Адоптивная (sA)":
            # Генерируем дни на лету: [20, 40, 60...]
            days = [t_start + i * t_interval for i in range(t_count)]
            self.active_injection_days = days

            # Функция проверки: попадает ли текущее время t в окно инъекции (1 сутки)
            v4 = lambda t: sA_amp if any(d <= t <= d + 1 for d in days) else 0.0

        # 3. Ингибирование
        elif mode == "Ингибирование (ηc, ημ)":
            v1 = lambda t: ec_amp
            v2 = lambda t: em_amp

        # 4. Комбинированная
        elif mode == "Комбинированная":
            days = [t_start + i * t_interval for i in range(t_count)]
            self.active_injection_days = days
            v1 = lambda t: ec_amp * 0.5
            v2 = lambda t: em_amp * 0.5
            v3 = lambda t: sC_amp if any(d <= t <= d + 1 for d in days) else 0.0
            v4 = lambda t: sA_amp if any(d <= t <= d + 1 for d in days) else 0.0

        return v1, v2, v3, v4

    def get_slider_value(self, name):
        slider, multiplier = self.sliders[name]
        return slider.value() / multiplier

    def update_label_text(self, key):
        slider, multiplier = self.sliders[key]
        val = slider.value() / multiplier
        self.value_labels[key].setText(self._format_value(key, val))

    def _set_slider_value(self, name, value):
        slider, multiplier = self.sliders[name]
        slider.setValue(int(value * multiplier))

    def _calculate_error(self, sol):
        clinical_times = self.clinical_data["time_days"].values
        clinical_T = self.clinical_data["linear_cells"].values

        # Интерполируем решение модели на точки времени из данных
        f_interp = interp1d(
            sol.t, sol.y[0], bounds_error=False, fill_value=(sol.y[0, 0], sol.y[0, -1])
        )
        T_pred = f_interp(clinical_times)

        # Считаем ошибку в логарифмической шкале (так лучше для экспоненциальных процессов)
        mask = (clinical_T > 0) & (T_pred > 0)
        if np.sum(mask) < 2:
            return 1e10

        mse = np.mean((np.log10(clinical_T[mask]) - np.log10(T_pred[mask])) ** 2)
        return mse

    def _cost_function(self, log_params, static_params, y0, t_span):
        c_val = 10 ** log_params[0]
        d_val = 10 ** log_params[1]

        # Подготавливаем функции терапии (обычно при фитинге они нулевые)
        v1, v2, v3, v4 = self.get_therapy_functions()

        try:
            sol = solve_ivp(
                tic_ode_system,
                t_span,
                y0,
                args=(
                    static_params["a"],
                    static_params["b"],
                    c_val,
                    static_params["mu"],
                    d_val,
                    static_params["p"],
                    static_params["lmbda"],
                    v1,
                    v2,
                    v3,
                    v4,
                ),
                method="BDF",
                vectorized=True,
            )
            if not sol.success:
                return 1e10

            error = self._calculate_error(sol)

            # print(f"DEBUG: c={c_val:.2e} d={d_val:.2e} -> MSE={error:.6f}")

            return error

        except Exception:
            return 1e10

    def _fit_parameters(self):
        if self.clinical_data is None or len(self.clinical_data) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет клинических данных!")
            return

        # --- ДОБАВЛЕНО ДЛЯ ДЕБАГА ---
        self.fit_call_count = 0
        print("\n--- Запуск оптимизации ---")
        # ----------------------------

        previous_therapy = self.therapy_combo.currentText()
        static = {p[0]: self.get_slider_value(p[0]) for p in self.param_list}
        y0 = [
            self.get_slider_value("T0"),
            self.get_slider_value("I0"),
            self.get_slider_value("C0"),
        ]
        t_span = (0, self.get_slider_value("t_end"))

        # В differential_evolution передаем дополнительные аргументы через args
        result = differential_evolution(
            self._cost_function,
            bounds=[
                (np.log10(1e-10), np.log10(1e-5)),
                (np.log10(1e-10), np.log10(1e-5)),
            ],
            args=(static, y0, t_span),
            popsize=15,
            workers=1,
            callback=self._optimization_callback,
            polish=True,  # Дополнительная локальная оптимизация в конце
        )
        optimal_c = 10 ** result.x[0]
        optimal_d = 10 ** result.x[1]
        final_cost = result.fun
        print("=" * 80)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"c = {optimal_c:.3e}")
        print(f"d = {optimal_d:.3e}")
        print(f"Final Cost = {final_cost:.5f}")
        print("=" * 80)
        # Возвращаем предыдущий тип терапии
        self.therapy_combo.setCurrentText(previous_therapy)
        self._set_slider_value("c", optimal_c)
        self._set_slider_value("d", optimal_d)
        self.update_plot()

    def _optimization_callback(self, xk, convergence=None):
        """
        xk: текущий лучший вектор параметров [log10(c), log10(d)]
        convergence: коэффициент сходимости (от 0 до 1)
        """
        self.fit_call_count += 1
        c_current = 10 ** xk[0]
        d_current = 10 ** xk[1]

        print(
            f"Итерация {self.fit_call_count:03d} | "
            f"Текущие параметры: c={c_current:.3e}, d={d_current:.3e} | "
            f"Сходимость: {convergence:.2%}"
        )

    def update_plot(self):
        # 1. Получаем текущие значения параметров со слайдеров
        params = [self.get_slider_value(p[0]) for p in self.param_list]
        y0 = [
            self.get_slider_value("T0"),
            self.get_slider_value("I0"),
            self.get_slider_value("C0"),
        ]
        t_span = (0, self.get_slider_value("t_end"))

        # 2. Запуск симуляции
        t_values, y_values = self.run_simulation(params, y0, t_span)

        if np.any(np.isnan(y_values)):
            print("WARNING: ODE solver produced NaN. Check therapy doses or tolerances.")
            self.auc_result_label.setText("Ошибка расчета (NaN)")
            return

        T, I = y_values[:, 0], y_values[:, 1]

        start_day = self.start_day_input.value()
        interval = self.interval_input.value()
        count = self.count_input.value()

        self.active_injection_days = [start_day + i * interval for i in range(count)]

        # 3. Очистка осей перед перерисовкой
        self.sim_canvas.axes.cla()

        # 4. Расчет AUC
        target_day = self.auc_day_input.value()
        mask = t_values <= target_day
        t_auc = t_values[mask]
        T_auc = T[mask]

        if len(t_auc) > 1:
            auc_value = np.trapezoid(T_auc, t_auc)
            auc_log_value = np.trapezoid(np.log10(np.maximum(T_auc, 1.0)), t_auc)
            self.auc_result_label.setText(f"AUC: {auc_value:.2e}")
            self.auc_result_label.setToolTip(f"Log10 AUC: {auc_log_value:.2f}")
            self.sim_canvas.axes.axvline(x=target_day, color="gray", linestyle=":", alpha=0.5)
        else:
            self.auc_result_label.setText("AUC: (вне диапазона)")

        # 5. Отрисовка модельных кривых
        self.sim_canvas.axes.plot(
            t_values, T, label="Опухолевые клетки (модель)", linewidth=3, linestyle="-"
        )
        self.sim_canvas.axes.plot(
            t_values, I, label="Иммунные клетки (модель)", linewidth=3, linestyle="--"
        )

        # 6. Отрисовка меток терапии (инъекций)
        mode = self.therapy_combo.currentText()
        if mode != "Без терапии":
            # Линия для легенды (рисуем один раз)
            self.sim_canvas.axes.plot(
                [], [], color="blue", linestyle="--", alpha=0.3, label="Инъекция"
            )
            # Реальные моменты впрыска
            for day in self.active_injection_days:
                if day <= t_span[1]:
                    self.sim_canvas.axes.axvline(x=day, color="blue", linestyle="--", alpha=0.2)

        # 7. Обработка клинических данных и расчет R^2
        if self.show_clinical_cb.isChecked() and self.clinical_data is not None:
            clinical_times = self.clinical_data["time_days"].values
            clinical_T = self.clinical_data["linear_cells"].values

            # Отрисовка клинических точек
            self.sim_canvas.axes.errorbar(
                clinical_times,
                clinical_T,
                yerr=self.clinical_data_yerr,
                fmt="o",
                color="red",
                label="Клинические данные",
                markersize=7,
                capsize=5,
                alpha=0.7,
            )

            # Расчет R^2 через интерполяцию модели на время клиники
            f_interp = interp1d(t_values, T, bounds_error=False, fill_value=(T[0], T[-1]))
            T_pred = f_interp(clinical_times)

            r2_text = "N/A"
            valid = (clinical_T > 0) & (T_pred > 0)
            if np.sum(valid) > 2:
                slope, intercept, r_value, p_value, std_err = linregress(
                    np.log10(clinical_T[valid]), np.log10(T_pred[valid])
                )
                r2_text = f"{r_value**2:.3f}"

            # 8. Отображение плашки R^2 (ТЕПЕРЬ ВНУТРИ УСЛОВИЯ)
            self.sim_canvas.axes.text(
                0.02,
                0.98,
                f"$R^2 = {r2_text}$",
                transform=self.sim_canvas.axes.transAxes,
                fontsize=20,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

        # 9. Динамическая настройка лимитов Y (для логарифмической шкалы)
        all_y_data = []
        if len(y_values) > 0:
            T_vals, I_vals = y_values[:, 0], y_values[:, 1]
            all_y_data.extend(T_vals[T_vals > 0])
            all_y_data.extend(I_vals[I_vals > 0])

        if self.show_clinical_cb.isChecked() and self.clinical_data is not None:
            c_vals = self.clinical_data["linear_cells"].values
            all_y_data.extend(c_vals[c_vals > 0])

        if all_y_data:
            y_min, y_max = np.min(all_y_data), np.max(all_y_data)
            self.sim_canvas.axes.set_ylim(y_min * 0.5, y_max * 2.0)

        # 10. Финальное оформление
        self.sim_canvas.axes.set_xlabel("Время [дни]")
        self.sim_canvas.axes.set_ylabel("Концентрация клеток [кл/мл]")
        self.sim_canvas.axes.legend(loc="lower right")
        self.sim_canvas.axes.grid(True, which="both", alpha=0.3)
        self.sim_canvas.axes.set_yscale("log")

        # Обновление холста
        self.sim_canvas.fig.tight_layout()
        self.sim_canvas.draw()

    def _run_analysis(self):
        # 1. Сразу блокируем интерфейс
        self.analysis_run_button.setEnabled(False)
        self.analysis_run_button.setText("Выполняется...")
        # Позволяем Qt перерисовать кнопку (сделать её серой) перед тяжелым циклом
        QApplication.processEvents()

        try:
            # Создаем карту параметров один раз для всех режимов
            param_map = {p[1]: p[0] for p in self.param_list + self.init_list}
            metric_display_name = self.analysis_metric_combo.currentText()
            metric_key = self.metrics_map[metric_display_name]

            start_mult = self.analysis_range_start.value()
            end_mult = self.analysis_range_end.value()
            num_steps = self.analysis_steps.value()

            if self.analysis_mode_combo.currentText() == "Семейство кривых (1D)":
                param_name = self.analysis_param_combo.currentText()
                param_key = param_map[param_name]

                results, param_values = self._run_parameter_sweep(
                    param_key, start_mult, end_mult, num_steps
                )

                metric_values = [
                    self._calculate_metric(res["t"], res["T"], metric_key) for res in results
                ]

                self._plot_family_of_curves(
                    results, param_key, param_values, metric_values, metric_display_name
                )

            else:
                # Логика для Heatmap (2D)
                p1_name = self.analysis_param_combo.currentText()
                p2_name = self.analysis_param_combo2.currentText()
                p1_key, p2_key = param_map[p1_name], param_map[p2_name]

                # Расчет диапазонов
                range1 = [
                    self.get_slider_value(p1_key) * start_mult,
                    self.get_slider_value(p1_key) * end_mult,
                ]
                range2 = [
                    self.get_slider_value(p2_key) * start_mult,
                    self.get_slider_value(p2_key) * end_mult,
                ]

                p1_vals, p2_vals, z_matrix = self._run_2d_analysis(
                    p1_key, range1, p2_key, range2, steps=num_steps
                )

                self.plot_heatmap(p1_vals, p2_vals, z_matrix, p1_key, p2_key)

        except Exception as e:
            # Важно: показываем пользователю, что именно пошло не так
            QMessageBox.critical(self, "Ошибка анализа", f"Критическая ошибка: {str(e)}")

        finally:
            # 4. В любой ситуации (ошибка или успех) возвращаем кнопку в строй
            self.analysis_run_button.setEnabled(True)
            self.analysis_run_button.setText("Запустить анализ")

    def _run_parameter_sweep(self, param_key, start_multiplier, end_multiplier, num_steps):
        base_params = {p[0]: self.get_slider_value(p[0]) for p in self.param_list}
        init_params = {p[0]: self.get_slider_value(p[0]) for p in self.init_list}

        # Определяем, варьируем мы параметр или начальное условие
        is_init_param = param_key in init_params
        base_value = init_params[param_key] if is_init_param else base_params[param_key]
        # param_values = np.linspace(
        #     base_value * start_multiplier, base_value * end_multiplier, num_steps
        # )
        start_val = base_value * start_multiplier
        end_val = base_value * end_multiplier
        param_values = np.logspace(np.log10(start_val), np.log10(end_val), num_steps)
        results = []

        for value in param_values:
            curr_p = base_params.copy()
            curr_i = init_params.copy()

            if is_init_param:
                curr_i[param_key] = value
            else:
                curr_p[param_key] = value

            params_tuple = tuple(curr_p[p[0]] for p in self.param_list)
            y0 = [curr_i["T0"], curr_i["I0"], curr_i["C0"]]
            t_span = (0, curr_i["t_end"])

            t_values, y_values = self.run_simulation(params_tuple, y0, t_span)
            results.append({"param_value": value, "t": t_values, "T": y_values[:, 0]})

        return results, param_values

    def plot_heatmap(self, x_vals, y_vals, z_matrix, p1_key, p2_key):
        """Отрисовка тепловой карты (Heatmap)"""
        # Создаем локальную переменную для удобства, чтобы не писать длинный путь
        ax = self.analysis_canvas.axes
        ax.cla()

        norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=0, vmax=max(1.1, np.max(z_matrix)))

        X, Y = np.meshgrid(x_vals, y_vals)
        mesh = ax.pcolormesh(X, Y, z_matrix, shading="auto", cmap="RdYlGn_r", norm=norm)

        # Устанавливаем логарифмический масштаб ДО настройки аспектов
        ax.set_xscale("log")
        ax.set_yscale("log")

        # ДЕЛАЕМ ГРАФИК КВАДРАТНЫМ
        # set_box_aspect(1) делает саму рамку осей квадратной независимо от данных
        ax.set_box_aspect(1)

        ax.set_xlabel(f"Параметр {p1_key} [мл/(клетка·день)]")
        ax.set_ylabel(f"Параметр {p2_key} [мл/(клетка·день)]")

        # Управление цветовой шкалой
        if self.colorbar_ax is not None:
            try:
                self.colorbar_ax.remove()
            except Exception:
                pass
            self.colorbar_ax = None

        # Привязываем colorbar к конкретному ax, чтобы он не "ломал" пропорции основного окна
        self.colorbar_ax = self.analysis_canvas.fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

        metric_name = self.analysis_metric_combo.currentText()
        self.colorbar_ax.set_label(metric_name)

        self.analysis_canvas.draw()

    def _run_2d_analysis(self, p1_key, p1_range, p2_key, p2_range, steps=15):
        AUC_LIMIT = 2.58e10

        p1_vals = np.logspace(np.log10(p1_range[0]), np.log10(p1_range[1]), steps)
        p2_vals = np.logspace(np.log10(p2_range[0]), np.log10(p2_range[1]), steps)

        z_matrix = np.zeros((steps, steps))

        current_params = {
            p[0]: self.get_slider_value(p[0]) for p in self.param_list + self.init_list
        }
        metric_key = self.metrics_map[self.analysis_metric_combo.currentText()]

        for i, v2 in enumerate(p2_vals):
            for j, v1 in enumerate(p1_vals):
                current_params[p1_key] = v1
                current_params[p2_key] = v2

                p_list = [current_params[p[0]] for p in self.param_list]
                y0 = [current_params["T0"], current_params["I0"], current_params["C0"]]
                t_span = (0, current_params["t_end"])

                t, y = self.run_simulation(p_list, y0, t_span)

                val = self._calculate_metric(t, y[:, 0], metric_key)

                # ИСПРАВЛЕННАЯ ЛОГИКА:
                if metric_key == "auc":
                    # Линейное отношение. 0 - нет опухоли, 1 - достигнут лимит.
                    z_matrix[i, j] = val / AUC_LIMIT
                else:
                    z_matrix[i, j] = val

        return p1_vals, p2_vals, z_matrix

    def _plot_family_of_curves(self, results, param_key, param_values, metric_values, metric_label):
        self.analysis_canvas.fig.clear()

        # Создаем сетку: верхний график (динамика), нижний (метрика) и цветовая шкала
        gs = self.analysis_canvas.fig.add_gridspec(
            2, 2, width_ratios=[25, 1], hspace=0.4, wspace=0.15
        )

        ax1 = self.analysis_canvas.fig.add_subplot(gs[0, 0])
        ax2 = self.analysis_canvas.fig.add_subplot(gs[1, 0])
        cax = self.analysis_canvas.fig.add_subplot(gs[:, 1])

        colors = cm.viridis_r(np.linspace(0, 1, len(results)))

        # 1. Верхний график: Семейство кривых
        for i, res in enumerate(results):
            ax1.plot(res["t"], res["T"], color=colors[i], alpha=0.6, linewidth=1.5)

        ax1.set_yscale("log")
        ax1.set_ylabel("Концентрация T")
        # ax1.set_title(f"Влияние параметра {param_key} на динамику")
        ax1.grid(True, which="both", alpha=0.2)

        # 2. Нижний график: Пузырьковая диаграмма (Bubble Chart)
        # Если выбрана AUC, делаем акцент на размере кружков
        metric_display = np.array(metric_values)

        # Масштабируем размеры кружков (S соответствует площади в пунктах^2)
        # Подбираем коэффициент так, чтобы кружки были видны, но не перекрывали всё
        if "AUC" in metric_label or "Интегральная" in metric_label:
            # Нормализуем значения для размеров (от 20 до 500 единиц площади)
            if metric_display.max() != metric_display.min():
                sizes = 20 + 480 * (metric_display - metric_display.min()) / (
                    metric_display.max() - metric_display.min()
                )
            else:
                sizes = [100] * len(metric_display)
        else:
            sizes = [60] * len(metric_display)  # Обычный размер для других метрик

        # Рисуем линию (тренд)
        ax2.plot(
            param_values,
            metric_values,
            color="teal",
            alpha=0.3,
            linestyle="--",
            zorder=1,
        )

        # Рисуем сами "пузырьки"
        scatter = ax2.scatter(
            param_values,
            metric_values,
            s=sizes,
            c=metric_values,  # Цвет тоже зависит от значения
            cmap="YlOrRd",  # Теплые цвета для большой нагрузки
            edgecolors="black",
            linewidths=0.8,
            alpha=0.8,
            zorder=2,
        )

        ax2.set_xlabel(f"Значение параметра {param_key} [мл/(клетка·день)]")
        ax2.set_ylabel(metric_label)
        ax2.grid(True, alpha=0.3)

        # 3. Настройка цветовой шкалы (справа)
        norm = cm.colors.Normalize(vmin=param_values.min(), vmax=param_values.max())
        sm = cm.ScalarMappable(cmap=cm.viridis_r, norm=norm)
        sm.set_array([])
        cbar = self.analysis_canvas.fig.colorbar(sm, cax=cax)
        cbar.set_label(f"Диапазон значений \nпараметра {param_key} [мл/(клетка·день)]")

        self.analysis_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
