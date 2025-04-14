# python_src/gui_app.py
import sys
import os
import numpy as np
import traceback
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QStatusBar, QFileDialog, QMessageBox, QSizePolicy, QTabWidget,
    QProgressBar, QSpacerItem, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsTextItem
)
from PyQt5.QtCore import Qt, QThreadPool, pyqtSlot, QTimer, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPen, QBrush, QColor, QPainter, QCursor, QTransform

from . import utils
from .gui_worker import TrainingWorker

try:
    import andreinet_bindings as anet
except ImportError:
    print("FATAL ERROR: gui_app.py - Could not import 'andreinet_bindings'.")
    print("Ensure C++ bindings are compiled and accessible (run setup.py).")
    sys.exit(1)


APP_STYLESHEET = """
    QWidget { font-size: 10pt; }
    QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 0.5em; }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px 0 3px; margin-left: 5px; }
    QPushButton { padding: 5px 10px; min-width: 80px; }
    QTextEdit { background-color: #f8f8f8; border: 1px solid #d0d0d0; }
    QProgressBar { text-align: center; }
    QLabel#StatusLabel { font-style: italic; color: #555; }
    QLabel#ErrorLabel { color: red; font-weight: bold; }
    QGraphicsView { border: 1px solid #c0c0c0; background-color: #ffffff; }
"""

class NetworkGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_panning = False; self._last_pan_pos = QPointF()
        self.setRenderHint(QPainter.Antialiasing); self.setRenderHint(QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse); self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self._min_zoom_factor = 0.1; self._max_zoom_factor = 10.0; self._current_zoom_factor = 1.0
    def wheelEvent(self, event):
        angle = event.angleDelta().y(); factor = 1.15 if angle > 0 else 1/1.15
        if angle > 0 and self._current_zoom_factor * factor > self._max_zoom_factor: factor = self._max_zoom_factor / self._current_zoom_factor
        elif angle < 0 and self._current_zoom_factor * factor < self._min_zoom_factor: factor = self._min_zoom_factor / self._current_zoom_factor
        if abs(factor - 1.0) > 1e-6: self.scale(factor, factor); self._current_zoom_factor *= factor
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton: self._is_panning = True; self._last_pan_pos = event.pos(); self.setCursor(Qt.ClosedHandCursor); self.setDragMode(QGraphicsView.ScrollHandDrag); event.accept()
        else: super().mousePressEvent(event)
    def mouseMoveEvent(self, event): super().mouseMoveEvent(event) # Base class handles ScrollHandDrag
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton: self._is_panning = False; self.setCursor(Qt.ArrowCursor); self.setDragMode(QGraphicsView.NoDrag); event.accept()
        else: super().mouseReleaseEvent(event)
    def reset_view(self):
        self.setTransform(QTransform()); self._current_zoom_factor = 1.0
        if self.scene() and self.scene().items(): self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)


class AndreiNetApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_net = None; self.raw_features = None; self.raw_targets = None; self.train_features = None; self.train_targets = None
        self.val_features = None; self.val_targets = None; self.train_data_cpp = None; self.val_data_cpp = None; self.scaler = None
        self.loaded_model_path = None; self.loaded_data_path = None; self.is_training = False; self.loss_figure = None; self.current_config = {}
        self.threadpool = QThreadPool(); print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")
        self.setWindowTitle("AndreiNET Visual Interface v1.3"); self.setGeometry(100, 100, 1500, 900)
        self.graphics_scene = QGraphicsScene(self)
        self._init_ui(); self.setStyleSheet(APP_STYLESHEET)

    def _init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        self.setup_tab = QWidget(); self.config_tab = QWidget(); self.train_eval_tab = QWidget()
        self.tabs.addTab(self.setup_tab, "  1. Setup & Data  "); self.tabs.addTab(self.config_tab, "  2. Configuration  "); self.tabs.addTab(self.train_eval_tab, "  3. Training & Evaluation  ")
        self._create_setup_tab(); self._create_config_tab(); self._create_train_eval_tab()
        output_frame = QFrame(); output_frame.setFrameShape(QFrame.StyledPanel); output_layout = QHBoxLayout(output_frame)
        output_layout.addWidget(self._create_structure_panel(), 2); output_layout.addWidget(self._create_log_panel(), 3)
        main_layout.addWidget(output_frame)
        self.statusBar = QStatusBar(); self.setStatusBar(self.statusBar); self.statusBar.showMessage("Welcome! Create/load network and load data.")
        self._update_ui_state(); self._draw_network_graph()

    # ==================================
    #       UI Tab/Panel Creation
    # ==================================
    def _create_setup_tab(self):
        # ... (Setup tab layout remains the same as previous version) ...
        main_layout = QHBoxLayout(self.setup_tab); left_col_widget = QWidget(); left_layout = QVBoxLayout(left_col_widget); left_layout.setAlignment(Qt.AlignTop)
        net_group = QGroupBox("Network Management"); net_grid_layout = QGridLayout(); self.topology_input = QLineEdit("2,4,1"); self.topology_input.setToolTip("Enter layer sizes, comma-separated (e.g., 2,4,1)")
        self.create_button = QPushButton("Create New Network"); self.create_button.setToolTip("Create a new network with the specified topology"); self.create_button.clicked.connect(self.create_network)
        self.load_button = QPushButton("Load Network File (.bin)"); self.load_button.setToolTip("Load weights into a created network (topology must match)"); self.load_button.clicked.connect(self.load_network)
        self.save_button = QPushButton("Save Network File (.bin)"); self.save_button.setToolTip("Save the current network's weights"); self.save_button.clicked.connect(self.save_network)
        net_grid_layout.addWidget(QLabel("Topology:"), 0, 0); net_grid_layout.addWidget(self.topology_input, 0, 1); net_grid_layout.addWidget(self.create_button, 1, 0, 1, 2); net_grid_layout.addWidget(self.load_button, 2, 0); net_grid_layout.addWidget(self.save_button, 2, 1)
        net_group.setLayout(net_grid_layout); left_layout.addWidget(net_group); left_layout.addStretch()
        right_col_widget = QWidget(); right_layout = QVBoxLayout(right_col_widget); right_layout.setAlignment(Qt.AlignTop)
        data_load_group = QGroupBox("Data Loading"); data_load_layout = QVBoxLayout()
        data_file_buttons = QHBoxLayout(); self.load_xor_button = QPushButton("Load XOR Preset"); self.load_xor_button.setToolTip("Load the standard 4-sample XOR dataset"); self.load_xor_button.clicked.connect(self.load_xor_data)
        self.load_csv_button = QPushButton("Load from CSV File"); self.load_csv_button.setToolTip("Load data from a CSV (last column is target)"); self.load_csv_button.clicked.connect(self.load_csv_dialog)
        data_file_buttons.addWidget(self.load_xor_button); data_file_buttons.addWidget(self.load_csv_button); data_load_layout.addLayout(data_file_buttons)
        csv_options_layout = QHBoxLayout(); csv_options_layout.addWidget(QLabel(" Delimiter:")); self.csv_delimiter_input = QLineEdit(","); self.csv_delimiter_input.setFixedWidth(40); self.csv_delimiter_input.setAlignment(Qt.AlignCenter)
        self.csv_header_checkbox = QCheckBox("Has Header Row"); csv_options_layout.addWidget(self.csv_delimiter_input); csv_options_layout.addWidget(self.csv_header_checkbox); csv_options_layout.addStretch(); data_load_layout.addLayout(csv_options_layout)
        self.data_status_label = QLabel("Status: No data loaded."); self.data_status_label.setObjectName("StatusLabel"); self.data_status_label.setWordWrap(True); data_load_layout.addWidget(self.data_status_label)
        data_load_group.setLayout(data_load_layout); right_layout.addWidget(data_load_group)
        self.data_prep_group = QGroupBox("Data Preprocessing"); self.data_prep_group.setEnabled(False)
        prep_layout = QFormLayout(); self.norm_combo = QComboBox(); self.norm_combo.addItems(["None", "MinMax (0 to 1)", "MinMax (-1 to 1)"]); self.norm_combo.setToolTip("Normalize feature values (applied after splitting)")
        self.split_spin = QDoubleSpinBox(); self.split_spin.setRange(0.0, 0.95); self.split_spin.setValue(0.2); self.split_spin.setDecimals(2); self.split_spin.setSingleStep(0.05); self.split_spin.setToolTip("Percentage of data to hold out for validation (0 = use all for training)")
        self.split_shuffle_checkbox = QCheckBox("Shuffle Before Split"); self.split_shuffle_checkbox.setChecked(True); self.split_shuffle_checkbox.setToolTip("Randomly shuffle data before splitting into train/validation sets")
        self.prepare_data_button = QPushButton("Prepare Data (Split & Normalize)"); self.prepare_data_button.setToolTip("Apply splitting and normalization to the loaded data"); self.prepare_data_button.clicked.connect(self.prepare_data)
        self.prep_status_label = QLabel("Status: Data not prepared."); self.prep_status_label.setObjectName("StatusLabel"); self.prep_status_label.setWordWrap(True)
        prep_layout.addRow("Normalization:", self.norm_combo); prep_layout.addRow("Validation Split %:", self.split_spin); prep_layout.addRow("", self.split_shuffle_checkbox); prep_layout.addRow(self.prepare_data_button); prep_layout.addRow(self.prep_status_label)
        self.data_prep_group.setLayout(prep_layout); right_layout.addWidget(self.data_prep_group); right_layout.addStretch()
        main_layout.addWidget(left_col_widget, 1); main_layout.addWidget(right_col_widget, 1)

    def _create_config_tab(self):
        # ... (Config tab layout remains the same) ...
        layout = QVBoxLayout(self.config_tab); layout.setAlignment(Qt.AlignTop); self.config_tab.setEnabled(False); grid_layout = QGridLayout(); grid_layout.setSpacing(15)
        act_group = QGroupBox("Layer Activations"); act_v_layout = QVBoxLayout(); self.activation_layout_container = QVBoxLayout(); act_v_layout.addLayout(self.activation_layout_container); act_group.setLayout(act_v_layout); grid_layout.addWidget(act_group, 0, 0, 2, 1)
        core_config_group = QGroupBox("Core Configuration"); core_config_layout = QFormLayout(core_config_group); self.loss_combo = QComboBox(); self.loss_combo.addItems(anet.LossFunction.__members__.keys())
        self.optimizer_combo = QComboBox(); self.optimizer_combo.addItems(anet.OptimizerType.__members__.keys()); self.optimizer_combo.currentTextChanged.connect(self._update_adam_params_state)
        core_config_layout.addRow("Loss Function:", self.loss_combo); core_config_layout.addRow("Optimizer:", self.optimizer_combo); grid_layout.addWidget(core_config_group, 0, 1)
        reg_group = QGroupBox("Regularization & Decay"); reg_layout = QFormLayout(reg_group); self.l2_spin = QDoubleSpinBox(); self.l2_spin.setRange(0.0, 1.0); self.l2_spin.setValue(0.0); self.l2_spin.setDecimals(6); self.l2_spin.setSingleStep(0.00001); self.l2_spin.setToolTip("L2 Regularization strength (lambda). 0 disables.")
        self.decay_spin = QDoubleSpinBox(); self.decay_spin.setRange(0.0, 1.0); self.decay_spin.setValue(0.0); self.decay_spin.setDecimals(6); self.decay_spin.setSingleStep(0.00001); self.decay_spin.setToolTip("Learning Rate decay factor per epoch. 0 disables.")
        reg_layout.addRow("L2 Lambda:", self.l2_spin); reg_layout.addRow("LR Decay:", self.decay_spin); grid_layout.addWidget(reg_group, 1, 1)
        self.adam_group = QGroupBox("ADAM Parameters"); self.adam_group.setCheckable(False); adam_layout = QFormLayout(self.adam_group)
        self.adam_beta1_spin = QDoubleSpinBox(); self.adam_beta1_spin.setDecimals(3); self.adam_beta1_spin.setRange(0.0, 0.999); self.adam_beta1_spin.setValue(0.9); self.adam_beta2_spin = QDoubleSpinBox(); self.adam_beta2_spin.setDecimals(4); self.adam_beta2_spin.setRange(0.0, 0.9999); self.adam_beta2_spin.setValue(0.999); self.adam_epsilon_spin = QDoubleSpinBox(); self.adam_epsilon_spin.setDecimals(10); self.adam_epsilon_spin.setRange(1e-12, 1e-1); self.adam_epsilon_spin.setValue(1e-8); self.adam_epsilon_spin.setSingleStep(1e-9)
        adam_layout.addRow("Beta1:", self.adam_beta1_spin); adam_layout.addRow("Beta2:", self.adam_beta2_spin); adam_layout.addRow("Epsilon:", self.adam_epsilon_spin); grid_layout.addWidget(self.adam_group, 2, 1)
        layout.addLayout(grid_layout); apply_layout = QHBoxLayout(); apply_layout.addStretch(); self.apply_config_button = QPushButton("Apply Configuration to Network"); self.apply_config_button.setToolTip("Save these settings to the current network instance"); self.apply_config_button.clicked.connect(self.apply_configuration)
        apply_layout.addWidget(self.apply_config_button); apply_layout.addStretch(); layout.addLayout(apply_layout); layout.addStretch(1)

    def _create_train_eval_tab(self):
        """Create the UI elements for the Training & Evaluation tab."""
        main_layout = QHBoxLayout(self.train_eval_tab)
        train_widget = QWidget(); train_v_layout = QVBoxLayout(train_widget); train_v_layout.setAlignment(Qt.AlignTop)
        self.train_group = QGroupBox("Training Controls"); self.train_group.setToolTip("Configure and start the network training process"); train_grid_layout = QGridLayout(self.train_group); train_grid_layout.setSpacing(10)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 1000000); self.epochs_spin.setValue(1000); self.epochs_spin.setGroupSeparatorShown(True)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 100000); self.batch_spin.setValue(4); self.batch_spin.setToolTip("Adjust based on training data size")
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(1e-7, 10.0); self.lr_spin.setValue(0.01); self.lr_spin.setDecimals(5); self.lr_spin.setSingleStep(0.001)
        self.train_shuffle_checkbox = QCheckBox("Shuffle Training Data Each Epoch"); self.train_shuffle_checkbox.setChecked(True); self.train_shuffle_checkbox.setToolTip("Randomize training sample order every epoch (Recommended)")
        self.train_button = QPushButton("Start Training"); self.train_button.setStyleSheet("QPushButton { background-color: #c8e6c9; border: 1px solid #a5d6a7; } QPushButton:hover { background-color: #a5d6a7; }"); self.train_button.clicked.connect(self.start_training)
        self.plot_loss_button = QPushButton("Plot Last Training Loss"); self.plot_loss_button.setToolTip("Display the loss curve from the most recent training run"); self.plot_loss_button.clicked.connect(self.plot_loss); self.plot_loss_button.setEnabled(False)
        self.train_progress_bar = QProgressBar(); self.train_progress_bar.setRange(0, 0); self.train_progress_bar.setTextVisible(False); self.train_progress_bar.setVisible(False)
        train_grid_layout.addWidget(QLabel("Epochs:"), 0, 0); train_grid_layout.addWidget(self.epochs_spin, 0, 1); train_grid_layout.addWidget(QLabel("Batch Size:"), 1, 0); train_grid_layout.addWidget(self.batch_spin, 1, 1)
        train_grid_layout.addWidget(QLabel("Initial LR:"), 2, 0); train_grid_layout.addWidget(self.lr_spin, 2, 1); train_grid_layout.addWidget(self.train_shuffle_checkbox, 3, 0, 1, 2)
        train_grid_layout.addWidget(self.train_button, 4, 0); train_grid_layout.addWidget(self.plot_loss_button, 4, 1); train_grid_layout.addWidget(self.train_progress_bar, 5, 0, 1, 2)
        train_v_layout.addWidget(self.train_group); train_v_layout.addStretch()

        eval_widget = QWidget(); eval_v_layout = QVBoxLayout(eval_widget); eval_v_layout.setAlignment(Qt.AlignTop)
        self.eval_pred_group = QGroupBox("Evaluation & Prediction"); ep_layout = QVBoxLayout(self.eval_pred_group)
        self.eval_button = QPushButton("Evaluate on Validation Set"); self.eval_button.setToolTip("Calculate accuracy/loss on the held-out validation set (if created)"); self.eval_button.clicked.connect(self.evaluate_network); ep_layout.addWidget(self.eval_button)

        # --- ADDED Compare Buttons ---
        compare_layout = QHBoxLayout()
        self.compare_train_button = QPushButton("Compare Outputs (Train Set)")
        self.compare_train_button.setToolTip("Show predictions vs targets for the training data")
        self.compare_train_button.clicked.connect(self.compare_train_outputs) # New Slot
        self.compare_val_button = QPushButton("Compare Outputs (Val Set)")
        self.compare_val_button.setToolTip("Show predictions vs targets for the validation data")
        self.compare_val_button.clicked.connect(self.compare_val_outputs) # New Slot
        compare_layout.addWidget(self.compare_train_button)
        compare_layout.addWidget(self.compare_val_button)
        ep_layout.addLayout(compare_layout) # Add compare buttons below evaluate
        # --- END Added Compare Buttons ---

        ep_layout.addWidget(QLabel("Predict Single Input:")); pred_layout = QHBoxLayout(); self.predict_input = QLineEdit(); self.predict_input.setPlaceholderText("Enter comma-separated features for prediction...")
        self.predict_button = QPushButton("Predict"); self.predict_button.setToolTip("Predict output for the entered feature vector"); self.predict_button.clicked.connect(self.predict_single); pred_layout.addWidget(self.predict_input); pred_layout.addWidget(self.predict_button); ep_layout.addLayout(pred_layout)
        eval_v_layout.addWidget(self.eval_pred_group); eval_v_layout.addStretch()
        main_layout.addWidget(train_widget, 1); main_layout.addWidget(eval_widget, 1); self.train_eval_tab.setEnabled(False)


    def _create_structure_panel(self):
        # ... (Structure panel layout remains the same as previous version, using NetworkGraphicsView) ...
        struct_group = QGroupBox("Network Structure / Configuration"); main_v_layout = QVBoxLayout(struct_group)
        top_h_layout = QHBoxLayout(); self.struct_text = QTextEdit(); self.struct_text.setReadOnly(True); self.struct_text.setFont(QFont("Courier New", 9))
        self.struct_text.setPlainText("Create or load a network."); self.struct_text.setMinimumHeight(150); self.struct_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        top_h_layout.addWidget(self.struct_text, 1)
        struct_button_layout = QVBoxLayout(); self.refresh_struct_button = QPushButton("Refresh Text"); self.refresh_struct_button.setToolTip("Update the text structure display"); self.refresh_struct_button.clicked.connect(self.display_structure_text)
        self.reset_view_button = QPushButton("Reset View"); self.reset_view_button.setToolTip("Reset zoom/pan for network visualization"); self.reset_view_button.clicked.connect(self._reset_graphics_view)
        struct_button_layout.addWidget(self.refresh_struct_button); struct_button_layout.addWidget(self.reset_view_button); struct_button_layout.addStretch(); top_h_layout.addLayout(struct_button_layout, 0)
        main_v_layout.addLayout(top_h_layout)
        graph_group = QGroupBox("Network Visualization"); graph_layout = QVBoxLayout(graph_group)
        self.graphics_view = NetworkGraphicsView(); self.graphics_view.setScene(self.graphics_scene); self.graphics_view.setMinimumHeight(250); self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        graph_layout.addWidget(self.graphics_view); main_v_layout.addWidget(graph_group)
        return struct_group

    def _create_log_panel(self):
        # ... (Log panel layout remains the same) ...
        results_group = QGroupBox("Log & Results"); results_layout = QVBoxLayout(results_group); self.results_text = QTextEdit(); self.results_text.setReadOnly(True); self.results_text.setFont(QFont("Courier New", 9))
        self.results_text.setPlaceholderText("Status, training logs, evaluation results, predictions..."); self.results_text.setMinimumHeight(150); clear_log_button = QPushButton("Clear Log"); clear_log_button.setToolTip("Clear the content of this log area"); clear_log_button.clicked.connect(self.results_text.clear)
        button_layout = QHBoxLayout(); button_layout.addStretch(); button_layout.addWidget(clear_log_button); results_layout.addLayout(button_layout); results_layout.addWidget(self.results_text); return results_group


    # ==================================
    #       Network Visualization
    # ==================================
    # ... (_draw_network_graph, resizeEvent, _refit_graphics_view, _reset_graphics_view remain the same) ...
    def _draw_network_graph(self):
        self.graphics_scene.clear();
        if not self.current_net: text = self.graphics_scene.addText("No Network Loaded", QFont("Arial", 12)); text.setDefaultTextColor(QColor("gray")); view_rect = self.graphics_view.viewport().rect(); text_rect = text.boundingRect(); center_x = max(0, (view_rect.width() - text_rect.width()) / 2); center_y = max(0, (view_rect.height() - text_rect.height()) / 2); text.setPos(center_x, center_y); self.graphics_view.setSceneRect(QRectF(0, 0, max(200, view_rect.width()), max(100, view_rect.height()))); self._refit_graphics_view(); return
        try:
            num_layers = self.current_net.get_layer_count(); layer_sizes = [self.current_net.get_layer_nodes(i) for i in range(num_layers)]; assert layer_sizes
            node_radius = 12; node_diam = node_radius * 2; h_spacing = 100; v_spacing_node = 15; layer_label_font = QFont("Arial", 9); label_metrics = QFontMetrics(layer_label_font); label_height = label_metrics.height() * 2.2; label_margin = 10; top_margin = label_margin + label_height; bottom_margin = 30; left_margin = 30; right_margin = 30
            max_nodes_in_layer = max(layer_sizes) if layer_sizes else 0; max_layer_content_height = max(1, max_nodes_in_layer) * node_diam + max(0, max_nodes_in_layer - 1) * v_spacing_node
            scene_height_needed = top_margin + max_layer_content_height + bottom_margin; scene_width_needed = left_margin + max(0, num_layers - 1) * h_spacing + right_margin + node_diam
            node_pen = QPen(QColor("#333333")); node_brush = QBrush(QColor("#64b5f6")); input_brush = QBrush(QColor("#bdbdbd")); output_brush = QBrush(QColor("#e91e63")); line_pen = QPen(QColor("#9e9e9e"), 0.8); line_pen.setCapStyle(Qt.RoundCap)
            layer_x = left_margin + node_radius; node_positions = [[] for _ in range(num_layers)]
            for i in range(num_layers):
                num_nodes = layer_sizes[i]; layer_content_height = num_nodes * node_diam + max(0, num_nodes - 1) * v_spacing_node; start_y = top_margin + (max_layer_content_height - layer_content_height) / 2; current_y = start_y + node_radius
                for j in range(num_nodes): node_center_x = layer_x; node_center_y = current_y; node_positions[i].append((node_center_x, node_center_y)); brush = node_brush;
                if i == 0: brush = input_brush;
                elif i == num_layers - 1: brush = output_brush; ellipse = QGraphicsEllipseItem(node_center_x - node_radius, node_center_y - node_radius, node_diam, node_diam); ellipse.setPen(node_pen); ellipse.setBrush(brush); self.graphics_scene.addItem(ellipse); current_y += node_diam + v_spacing_node
                text = QGraphicsTextItem(f"L{i}\n({num_nodes}n)"); text.setFont(layer_label_font); text.setDefaultTextColor(QColor("#0d47a1")); text_rect = text.boundingRect(); text.setPos(layer_x - text_rect.width()/2, label_margin - 5) ; self.graphics_scene.addItem(text); layer_x += h_spacing # Position labels based on top margin
            for i in range(num_layers - 1):
                for x1, y1 in node_positions[i]:
                    for x2, y2 in node_positions[i+1]: line = QGraphicsLineItem(x1, y1, x2, y2); line.setPen(line_pen); line.setZValue(-1); self.graphics_scene.addItem(line)
            self.graphics_scene.setSceneRect(0, 0, scene_width_needed, scene_height_needed); self._refit_graphics_view()
        except Exception as e: print(f"Error drawing graph: {e}\n{traceback.format_exc()}"); self.graphics_scene.clear(); text = self.graphics_scene.addText(f"Error drawing graph:\n{e}", QFont("Arial", 10)); text.setDefaultTextColor(QColor("red"))

    def resizeEvent(self, event): super().resizeEvent(event); QTimer.singleShot(100, self._refit_graphics_view)
    def _refit_graphics_view(self):
        if self.graphics_scene and self.graphics_scene.itemsBoundingRect().isValid(): self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect().adjusted(-10, -10, 10, 10), Qt.KeepAspectRatio)
        elif self.graphics_scene: self.graphics_view.reset_view()
    def _reset_graphics_view(self):
        if hasattr(self, 'graphics_view') and isinstance(self.graphics_view, NetworkGraphicsView): self.graphics_view.reset_view(); self.log_message("Graphics view reset.")
        else: self.log_message("Graphics view not available.", is_error=True)

    # ==================================
    #       UI Update Logic & Slots
    # ==================================
    def _update_ui_state(self):
        # ... (Update logic including new compare buttons) ...
        net_exists = self.current_net is not None; raw_data_loaded = self.raw_features is not None and self.raw_targets is not None
        prepared_data_loaded = self.train_data_cpp is not None; val_data_exists = self.val_data_cpp is not None
        self.config_tab.setEnabled(net_exists); self.train_eval_tab.setEnabled(net_exists and prepared_data_loaded)
        self.save_button.setEnabled(net_exists); self.data_prep_group.setEnabled(raw_data_loaded)
        self.apply_config_button.setEnabled(net_exists)
        if net_exists: self._update_adam_params_state(); 
        else: self.adam_group.setEnabled(False)
        can_train = net_exists and prepared_data_loaded and not self.is_training
        can_eval = net_exists and val_data_exists and not self.is_training
        can_compare_train = net_exists and prepared_data_loaded and not self.is_training
        can_compare_val = net_exists and val_data_exists and not self.is_training
        self.train_group.setEnabled(net_exists and prepared_data_loaded); self.train_button.setText("Start Training" if not self.is_training else "Training..."); self.train_button.setEnabled(can_train)
        self.eval_pred_group.setEnabled(net_exists); self.eval_button.setEnabled(can_eval)
        self.compare_train_button.setEnabled(can_compare_train) # Enable compare train button
        self.compare_val_button.setEnabled(can_compare_val)     # Enable compare val button
        self.predict_input.setEnabled(net_exists and not self.is_training); self.predict_button.setEnabled(net_exists and not self.is_training)
        self.train_progress_bar.setVisible(self.is_training);
        if not self.is_training: self.train_progress_bar.setValue(0)
        self.refresh_struct_button.setEnabled(net_exists); self.reset_view_button.setEnabled(net_exists)

    # ... (_update_dynamic_activation_widgets, _update_adam_params_state, log_message remain the same) ...
    def _update_dynamic_activation_widgets(self):
        """Clear and recreate activation widgets in the Config tab."""
        # --- Robust Clearing Loop ---
        if self.activation_layout_container is not None: # Check if layout exists
             # Iterate backwards when removing items to avoid index issues
             for i in reversed(range(self.activation_layout_container.count())):
                 item = self.activation_layout_container.itemAt(i) # Get item without removing yet
                 if item is None:
                     continue # Skip if item is somehow None

                 widget = item.widget()
                 layout = item.layout()

                 if widget is not None:
                     # If it's a widget, remove it from layout and delete it
                     self.activation_layout_container.removeItem(item)
                     widget.deleteLater()
                 elif layout is not None:
                     # If it's a layout, clear it recursively and delete it
                     while layout.count():
                         child_item = layout.takeAt(0) # Remove child item
                         if child_item is None: continue

                         child_widget = child_item.widget()
                         child_layout = child_item.layout()

                         if child_widget is not None:
                             child_widget.deleteLater()
                         elif child_layout is not None:
                             # Simple recursive deletion for nested layouts (adjust if deeper nesting)
                             while child_layout.count():
                                 sub_child = child_layout.takeAt(0)
                                 if sub_child.widget(): sub_child.widget().deleteLater()
                             layout.removeItem(child_item) # Remove the layout item itself
                             child_layout.deleteLater() # Delete the layout object
                     # After clearing the inner layout, remove the layout item from the main container
                     self.activation_layout_container.removeItem(item)
                     layout.deleteLater() # Delete the layout object itself

        # --- End Robust Clearing Loop ---

        self.activation_widgets = [] # Reset the tracking list

        if not self.current_net: return # Exit if no network

        # --- Create new widgets (remains the same) ---
        num_layers = self.current_net.get_layer_count()
        for i in range(num_layers):
            layer_type = "Input" if i == 0 else "Output" if i == num_layers - 1 else f"Hidden {i}"
            nodes = self.current_net.get_layer_nodes(i)
            current_act_id = self.current_net.get_layer_activation(i)

            h_layout = QHBoxLayout() # Create a new layout for each layer's widgets
            label = QLabel(f"L{i} ({layer_type}, {nodes}n):")
            combo = QComboBox()
            combo.addItems(utils.ACTIVATION_MAP.values())
            if current_act_id in utils.ACTIVATION_MAP:
                 combo.setCurrentText(utils.ACTIVATION_MAP[current_act_id])
            else: combo.setCurrentIndex(0)
            combo.setToolTip(f"Activation function for Layer {i}")

            h_layout.addWidget(label)
            h_layout.addWidget(combo)
            # Add the HBox layout (containing label and combo) to the container VBox
            self.activation_layout_container.addLayout(h_layout)
            self.activation_widgets.append((label, combo)) # Store references

    def _update_adam_params_state(self):
        if not self.config_tab.isEnabled(): self.adam_group.setEnabled(False); return
        is_adam = self.optimizer_combo.currentText() == "ADAM"; self.adam_group.setEnabled(is_adam)

    def log_message(self, message, is_error=False, status_bar_time=5000):
        prefix = "[ERROR] " if is_error else "[INFO] "; message_str = str(message)
        self.results_text.append(prefix + message_str); self.results_text.verticalScrollBar().setValue(self.results_text.verticalScrollBar().maximum())
        if status_bar_time > 0: self.statusBar.showMessage(message_str, status_bar_time)
        print(prefix + message_str)

    # --- Actions ---
    # ... (create_network, load_network, save_network remain the same, calling self.display_structure()) ...
    def create_network(self):
        self.log_message("Attempting to create network..."); topo_str = self.topology_input.text().strip()
        try:
            topology = [int(n.strip()) for n in topo_str.split(',')]; assert len(topology) >= 2 and all(n > 0 for n in topology)
            self.current_net = anet.Net(topology); self.loaded_model_path = None; self._reset_data_state(); self.current_config = {}
            self.log_message(f"Network created: {topology}"); self._update_dynamic_activation_widgets(); QTimer.singleShot(10, self._apply_default_config); self.display_structure(); self.statusBar.showMessage(f"Network {topology} created.", 5000); self.plot_loss_button.setEnabled(False); self.results_text.clear()
        except Exception as e: self.log_message(f"Invalid topology or creation error: {e}", is_error=True); QMessageBox.warning(self, "Error", f"Invalid topology or creation error: {e}"); self.current_net = None; self.loaded_model_path = None
        finally: self._update_ui_state()

    def load_network(self):
        default_dir = os.path.join(os.getcwd(), "models"); os.makedirs(default_dir, exist_ok=True); options = QFileDialog.Options(); fileName, _ = QFileDialog.getOpenFileName(self, "Load Model", default_dir, "AndreiNET Models (*.bin);;All Files (*)", options=options)
        if fileName:
            if self.current_net is None: QMessageBox.warning(self, "Load Error", "Create network with matching topology first."); self.log_message("Load cancelled.", is_error=True); return
            self.log_message(f"Loading network from: {fileName}");
            try:
                self.current_net.load(fileName); self.loaded_model_path = fileName; self._reset_data_state(); self.current_config = {}
                self.log_message("Network loaded! Re-apply config if needed."); self.statusBar.showMessage(f"Loaded {os.path.basename(fileName)}.", 5000); self._update_dynamic_activation_widgets(); self.display_structure(); self.plot_loss_button.setEnabled(False); self.results_text.clear()
            except Exception as e: self.log_message(f"Error loading: {e}\n{traceback.format_exc()}", is_error=True); QMessageBox.critical(self, "Load Error", f"Failed to load:\n{e}\n\nEnsure topology matches file."); self.loaded_model_path = None
            finally: self._update_ui_state()

    def save_network(self):
        if not self.current_net: self.log_message("No network.", is_error=True); return
        default_dir = os.path.join(os.getcwd(), "models"); os.makedirs(default_dir, exist_ok=True); suggested_name = os.path.basename(self.loaded_model_path) if self.loaded_model_path else "model.bin"; default_path = os.path.join(default_dir, suggested_name)
        options = QFileDialog.Options(); fileName, _ = QFileDialog.getSaveFileName(self, "Save Model", default_path, "AndreiNET Models (*.bin);;All Files (*)", options=options)
        if fileName:
            try: self.current_net.save(fileName); self.loaded_model_path = fileName; self.log_message(f"Network saved to: {fileName}"); self.statusBar.showMessage("Network saved.", 3000)
            except Exception as e: self.log_message(f"Error saving network: {e}\n{traceback.format_exc()}", is_error=True); QMessageBox.critical(self, "Save Error", f"Could not save: {e}")

    # _reset_data_state, load_xor_data, load_csv_dialog, prepare_data remain the same
    def _reset_data_state(self):
        self.raw_features = None; self.raw_targets = None; self.train_features = None; self.train_targets = None; self.val_features = None; self.val_targets = None
        self.train_data_cpp = None; self.val_data_cpp = None; self.scaler = None; self.loaded_data_path = None
        self.data_status_label.setText("Status: No data loaded."); self.prep_status_label.setText("Status: Data not prepared."); print("Data state reset.")
        self.plot_loss_button.setEnabled(False); self._update_ui_state()

    def load_xor_data(self):
        self.log_message("Loading XOR preset...");
        try:
            xor_cpp_data = utils.load_xor_data(); assert xor_cpp_data is not None
            self._reset_data_state(); self.train_data_cpp = xor_cpp_data; self.loaded_data_path = "XOR Preset"; num_samples = len(self.train_data_cpp)
            self.data_status_label.setText(f"Status: Loaded XOR ({num_samples} samples)."); self.prep_status_label.setText("Status: Using raw data (split=0, norm=None).")
            self.log_message(f"XOR loaded & prepared ({num_samples} samples)."); self.statusBar.showMessage("XOR data loaded.", 3000)
            self.batch_spin.setMaximum(num_samples); self.batch_spin.setValue(min(self.batch_spin.value(), num_samples))
        except Exception as e: self.log_message(f"Error loading XOR: {e}\n{traceback.format_exc()}", is_error=True); QMessageBox.warning(self, "Data Load Error", f"Could not load XOR: {e}"); self._reset_data_state()
        finally: self._update_ui_state()

    def load_csv_dialog(self):
        default_dir = os.path.join(os.getcwd(), "datasets"); os.makedirs(default_dir, exist_ok=True); options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Load CSV File", default_dir, "CSV Files (*.csv *.txt *.dat);;All Files (*)", options=options)
        if fileName:
            delimiter = self.csv_delimiter_input.text(); has_header = self.csv_header_checkbox.isChecked()
            self.log_message(f"Loading CSV: {fileName} (Delim:'{delimiter}', Hdr:{has_header})");
            try:
                self._reset_data_state(); self.raw_features, self.raw_targets = utils.load_csv_data(fileName, delimiter, has_header); self.loaded_data_path = fileName
                num_samples = self.raw_features.shape[0]; num_features = self.raw_features.shape[1]
                self.data_status_label.setText(f"Status: Loaded {num_samples}s, {num_features}f. Ready for Prep."); self.log_message(f"CSV loaded ({num_samples}s, {num_features}f)."); self.statusBar.showMessage("CSV data loaded. Prepare it.", 5000)
            except Exception as e: self.log_message(f"Error loading CSV: {e}\n{traceback.format_exc()}", is_error=True); QMessageBox.critical(self, "CSV Load Error", f"Failed to load CSV:\n{e}"); self._reset_data_state()
            finally: self._update_ui_state()

    def prepare_data(self):
        if self.raw_features is None: self.log_message("Load raw data first.", is_error=True); return
        self.log_message("Preparing data...");
        
        try:
            split_perc = self.split_spin.value(); shuffle_split = self.split_shuffle_checkbox.isChecked()
            X_train, y_train, X_val, y_val = utils.split_data(self.raw_features, self.raw_targets, validation_split=split_perc, shuffle=shuffle_split)
            self.train_features, self.train_targets = X_train, y_train; self.val_features, self.val_targets = X_val, y_val; val_count = X_val.shape[0] if X_val is not None else 0; self.log_message(f"Data split: {X_train.shape[0]} train, {val_count} validation.")
            norm_choice = self.norm_combo.currentText(); feature_range = (0, 1); self.scaler = None
            if norm_choice == "MinMax (-1 to 1)": feature_range = (-1, 1)
            if norm_choice != "None":
                 if utils.MinMaxScaler is None: raise RuntimeError("Cannot normalize: scikit-learn missing.")
                 if self.train_features is None: raise RuntimeError("Training features missing.")
                 X_train_scaled, X_val_scaled, scaler = utils.normalize_data(self.train_features, self.val_features, feature_range=feature_range)
                 if scaler is not None: self.train_features = X_train_scaled; self.val_features = X_val_scaled; self.scaler = scaler; self.log_message(f"Features normalized ({norm_choice}).")
                 else: self.log_message("Normalization step failed.", is_error=True)
            else: self.log_message("Normalization skipped.")
            if self.train_features is not None: self.train_data_cpp = utils.format_data_for_cpp(self.train_features, self.train_targets)
            else: self.train_data_cpp = None
            self.val_data_cpp = None
            if self.val_features is not None: self.val_data_cpp = utils.format_data_for_cpp(self.val_features, self.val_targets)
            train_cpp_count = len(self.train_data_cpp) if self.train_data_cpp else 0; val_cpp_count = len(self.val_data_cpp) if self.val_data_cpp else 0
            prep_info = f"Prepared: {train_cpp_count} train";
            if val_cpp_count > 0: prep_info += f", {val_cpp_count} val"
            prep_info += f". Norm: {norm_choice}."; self.prep_status_label.setText(f"Status: {prep_info}"); self.log_message(f"Data prep complete. {prep_info}"); self.statusBar.showMessage("Data prepared for training.", 3000)
            if self.train_data_cpp: self.batch_spin.setMaximum(len(self.train_data_cpp)); self.batch_spin.setValue(min(self.batch_spin.value(), len(self.train_data_cpp)))
            else: self.batch_spin.setMaximum(1); self.batch_spin.setValue(1)
        except Exception as e:
            self.log_message(f"Error preparing data: {e}\n{traceback.format_exc()}", is_error=True); QMessageBox.critical(self, "Data Prep Error", f"Failed to prepare data:\n{e}")
            self.train_features=None; self.train_targets=None; self.val_features=None; self.val_targets=None; self.train_data_cpp=None; self.val_data_cpp=None; self.scaler=None; self.prep_status_label.setText("Status: Preparation failed.")
        finally: self._update_ui_state()

    # apply_configuration, _apply_default_config remain the same
    def apply_configuration(self):
        if not self.current_net: self.log_message("No network.", is_error=True); return
        self.log_message("Applying config..."); 
        try:
            config = {}; num_layers = self.current_net.get_layer_count(); act_config = []
            if len(self.activation_widgets) == num_layers:
                 for i in range(num_layers): _, combo = self.activation_widgets[i]; act_name = combo.currentText(); act_id = utils.ACTIVATION_IDS.get(act_name.lower(), -1); self.current_net.set_layer_activation(i, act_id); act_config.append(act_name)
                 config['activations'] = act_config; self.log_message(f"Set activations: {'->'.join(act_config)}")
            loss_name = self.loss_combo.currentText(); loss_enum = getattr(anet.LossFunction, loss_name, anet.LossFunction.MSE); self.current_net.set_loss_function(loss_enum); config['loss'] = loss_name; self.log_message(f"Set loss: {loss_name}")
            opt_name = self.optimizer_combo.currentText(); opt_enum = getattr(anet.OptimizerType, opt_name, anet.OptimizerType.SGD); config['optimizer'] = opt_name
            if opt_enum == anet.OptimizerType.ADAM: b1 = self.adam_beta1_spin.value(); b2 = self.adam_beta2_spin.value(); eps = self.adam_epsilon_spin.value(); self.current_net.set_optimizer(opt_enum, b1, b2, eps); config['adam_params'] = {'b1': b1, 'b2': b2, 'eps': eps}; self.log_message(f"Set optimizer: ADAM (b1={b1:.3f}, b2={b2:.4f}, eps={eps:.1E})")
            else: self.current_net.set_optimizer(opt_enum); self.log_message("Set optimizer: SGD")
            l2 = self.l2_spin.value(); decay = self.decay_spin.value(); self.current_net.set_L2_regularization(l2); self.current_net.set_learning_rate_decay(decay); config['l2'] = l2; config['lr_decay'] = decay; self.log_message(f"Set L2={l2:.6f}, LR Decay={decay:.6f}")
            self.current_config = config; self.log_message("Configuration applied."); self.statusBar.showMessage("Configuration applied.", 3000); self.display_structure()
        except Exception as e: self.log_message(f"Error applying config: {e}\n{traceback.format_exc()}", is_error=True); QMessageBox.critical(self, "Error", f"Failed to apply config: {e}")
        finally: self._update_ui_state()

    def _apply_default_config(self):
        if not self.current_net: return
        try:
            self.loss_combo.setCurrentText("CROSS_ENTROPY"); self.optimizer_combo.setCurrentText("ADAM")
            num_layers = self.current_net.get_layer_count()
            if len(self.activation_widgets) == num_layers:
                for i in range(num_layers): _, combo = self.activation_widgets[i]; default_act = "Tanh" if 0 < i < num_layers - 1 else ("Sigmoid" if i == num_layers - 1 else "Linear"); combo.setCurrentText(default_act)
            self.apply_configuration()
        except Exception as e: print(f"Warning: Could not apply default config: {e}")

    # --- Display Logic ---
    # display_structure_text, display_structure remain the same
    def display_structure_text(self):
        if not self.current_net: self.struct_text.setPlainText("No network loaded."); return
        try:
            structure_str = self.current_net.get_network_structure_str(show_matrices=False); config_summary = "\n--- Current Configuration ---\n"; config_items = []
            if self.current_config:
                for key, value in self.current_config.items():
                    if key == 'adam_params': config_items.append(f"  ADAM Params: b1={value.get('b1', '?'):.3f}, b2={value.get('b2', '?'):.4f}, eps={value.get('eps', '?'):.1E}")
                    elif key == 'activations': config_items.append(f"  Activations: {' -> '.join(value)}")
                    elif isinstance(value, float): config_items.append(f"  {key.replace('_', ' ').title()}: {value:.6f}")
                    else: config_items.append(f"  {key.replace('_', ' ').title()}: {value}")
            else: config_items.append("(Default or not configured)")
            config_summary += "\n".join(config_items); self.struct_text.setPlainText(structure_str + config_summary); self.log_message("Network structure text updated.")
        except Exception as e: self.log_message(f"Error getting structure text: {e}", is_error=True); self.struct_text.setPlainText(f"[Error: {e}]")

    def display_structure(self): self.display_structure_text(); self._draw_network_graph()

    # --- Training & Evaluation ---
    # start_training, on_training_finished, on_training_error, on_training_status_update, plot_loss, evaluate_network, predict_single remain the same
    def start_training(self):
        if not self.current_net or not self.train_data_cpp or self.is_training:
            if self.is_training: msg = "Training in progress."
            elif not self.current_net: msg = "Create/load network."
            else: msg = "Load/prepare data."
            self.log_message(msg, is_error=True); QMessageBox.warning(self, "Cannot Start", msg); return
        self.is_training = True; self._update_ui_state(); self.train_progress_bar.setRange(0, 0); self.plot_loss_button.setEnabled(False); self.results_text.append("\n--- Starting Training ---"); self.statusBar.showMessage("Training started...")
        if not utils.clear_loss_log(): self.log_message("Could not clear loss log.", is_error=True)
        epochs = self.epochs_spin.value(); batch_size = self.batch_spin.value(); initial_lr = self.lr_spin.value(); shuffle = self.train_shuffle_checkbox.isChecked()
        worker = TrainingWorker(self.current_net, self.train_data_cpp, epochs, initial_lr, batch_size, shuffle, self.val_data_cpp); worker.signals.finished.connect(self.on_training_finished); worker.signals.error.connect(self.on_training_error); worker.signals.status_update.connect(self.on_training_status_update)
        self.threadpool.start(worker)

    @pyqtSlot()
    def on_training_finished(self):
        self.is_training = False; self._update_ui_state(); self.statusBar.showMessage("Training finished.", 5000); self.results_text.append("--- Training Complete ---")
        if os.path.exists("training_loss_eigen.txt"): self.plot_loss_button.setEnabled(True)
        else: self.log_message("Training finished, but loss log missing.", is_error=True)

    @pyqtSlot(tuple)
    def on_training_error(self, error_info):
        error_type, message = error_info; self.is_training = False; self._update_ui_state()
        self.log_message(f"Training Error ({error_type}): {message}", is_error=True); QMessageBox.critical(self, "Training Error", f"Error:\n({error_type}) {message}"); self.plot_loss_button.setEnabled(False)

    @pyqtSlot(str)
    def on_training_status_update(self, message): self.statusBar.showMessage(message, 3000); self.results_text.append(f"[TRAIN] {message}")

    def plot_loss(self):
         self.log_message("Plotting loss..."); fig = utils.plot_loss_history()
         if fig: self.loss_figure = fig; self.statusBar.showMessage("Loss plot shown.", 3000)
         else: self.log_message("Failed to plot loss.", is_error=True); QMessageBox.warning(self, "Plot Error", "Could not plot loss log.")

    def evaluate_network(self):
        if not self.current_net: self.log_message("No network.", is_error=True); return
        if not self.val_data_cpp: self.log_message("No validation data.", is_error=True); QMessageBox.information(self, "Evaluate", "No validation data. Prepare data with split > 0."); return
        self.log_message("Evaluating on validation set..."); self.results_text.append("\n--- Evaluation (Validation Set) ---"); correct = 0; total = len(self.val_data_cpp); output_size = self.current_net.get_layer_nodes(self.current_net.get_layer_count() - 1)
        try:
            for i, (features, target) in enumerate(self.val_data_cpp):
                prediction = self.current_net.predict(features); prediction_np = np.asarray(prediction); target_np = np.asarray(target); pred_str = ", ".join([f"{p:.4f}" for p in prediction_np]); result_str = ""
                if output_size == 1 and len(target_np) == 1: target_val = target_np[0]; pred_val = prediction_np[0]; pred_class = 1 if pred_val > 0.5 else 0; target_class = int(round(target_val)); is_correct = (pred_class == target_class); correct += is_correct; result_str = f" T={target_val:.1f} -> P={pred_val:.4f} (Cls:{pred_class}) {'[OK]' if is_correct else '[WRONG]'}"
                else: target_str = ", ".join([f"{t:.1f}" for t in target_np]); result_str = f" T=[{target_str}] -> P=[{pred_str}]"
                input_str = ", ".join([f"{f:.1f}" for f in np.asarray(features)]); self.results_text.append(f"S{i}: In=[{input_str}]{result_str}")
            accuracy = (correct / total) * 100 if total > 0 else 0; summary = f"Validation complete. Acc: {accuracy:.2f}% ({correct}/{total})"
            self.results_text.append(f"\n{summary}"); self.log_message(summary); self.statusBar.showMessage(summary, 5000)
        except Exception as e: self.log_message(f"Eval error: {e}\n{traceback.format_exc()}", is_error=True); QMessageBox.critical(self, "Evaluation Error", f"Error: {e}")

    # --- ADDED Slots for Compare Buttons ---
    @pyqtSlot()
    def compare_train_outputs(self):
        self._compare_outputs(dataset_type='train')

    @pyqtSlot()
    def compare_val_outputs(self):
        self._compare_outputs(dataset_type='validation')
    # --- END ADDED Slots ---

    # --- ADDED Compare Method ---
    def _compare_outputs(self, dataset_type='train'):
        """Displays Input -> Target -> Prediction for a chosen dataset."""
        if not self.current_net: self.log_message("No network loaded.", is_error=True); return

        data_list = None
        data_label = ""
        if dataset_type == 'train' and self.train_data_cpp:
            data_list = self.train_data_cpp
            data_label = "Training Set"
        elif dataset_type == 'validation' and self.val_data_cpp:
            data_list = self.val_data_cpp
            data_label = "Validation Set"

        if not data_list:
            self.log_message(f"No prepared {dataset_type} data available to compare.", is_error=True)
            QMessageBox.information(self, "Compare Outputs", f"No prepared {dataset_type} data available.")
            return

        self.log_message(f"Comparing network outputs for {data_label}...")
        self.results_text.append(f"\n--- Output Comparison ({data_label}) ---")
        output_size = self.current_net.get_layer_nodes(self.current_net.get_layer_count() - 1)

        try:
            line_format = "{:<5} | {:<25} | {:<15} | {:<15} | {:<8}"
            self.results_text.append(line_format.format("Idx", "Input Features", "Target", "Prediction", "Result"))
            self.results_text.append("-" * 75)

            for i, (features, target) in enumerate(data_list):
                prediction = self.current_net.predict(features)
                # Convert to numpy for easier handling/formatting
                features_np = np.asarray(features)
                target_np = np.asarray(target)
                prediction_np = np.asarray(prediction)

                # Format for display (limit precision)
                f_str = ", ".join([f"{f:.2f}" for f in features_np])
                t_str = ", ".join([f"{t:.2f}" for t in target_np])
                p_str = ", ".join([f"{p:.4f}" for p in prediction_np])

                # Add OK/WRONG for binary classification case
                result = ""
                if output_size == 1 and len(target_np) == 1:
                    pred_class = 1 if prediction_np[0] > 0.5 else 0
                    target_class = int(round(target_np[0]))
                    result = "[OK]" if pred_class == target_class else "[WRONG]"

                self.results_text.append(line_format.format(f"S{i}", f_str, t_str, p_str, result))

            self.log_message(f"Finished comparing outputs for {data_label}.")
            self.statusBar.showMessage(f"Displayed output comparison for {data_label}.", 4000)

        except Exception as e:
            self.log_message(f"Error during output comparison: {e}\n{traceback.format_exc()}", is_error=True)
            QMessageBox.critical(self, "Comparison Error", f"An error occurred during comparison: {e}")
    # --- END ADDED Compare Method ---

    # predict_single remains the same
    def predict_single(self):
        if not self.current_net: return
        input_str = self.predict_input.text().strip(); self.log_message(f"Predicting for input: '{input_str}'"); self.results_text.append(f"\n--- Single Prediction ---"); self.results_text.append(f"Input String: {input_str}")
        try:
            input_size = self.current_net.get_layer_nodes(0); features_list = [float(x.strip()) for x in input_str.split(',')]; assert len(features_list) == input_size
            features_np = np.array(features_list, dtype=np.float64).reshape(1, -1); scaled_features_np = features_np
            if self.scaler: scaled_features_np = self.scaler.transform(features_np); self.results_text.append(f"Normalized Input: {scaled_features_np.flatten()}")
            else: self.results_text.append(f"Input Features (Raw): {features_np.flatten()}")
            prediction = self.current_net.predict(scaled_features_np.flatten()); prediction_np = np.asarray(prediction)
            pred_str = ", ".join([f"{p:.5f}" for p in prediction_np]); self.results_text.append(f"Raw Prediction: [{pred_str}]")
            if len(prediction_np) == 1: pred_val = prediction_np[0]; pred_class = 1 if pred_val > 0.5 else 0; self.results_text.append(f"Interpreted Class (Threshold 0.5): {pred_class}")
            self.log_message(f"Prediction successful."); self.statusBar.showMessage("Prediction complete.", 3000)
        except ValueError as e: self.log_message(f"Invalid input: {e}", is_error=True); self.results_text.append(f"[ERROR] Invalid input: {e}"); QMessageBox.warning(self, "Invalid Input", f"Invalid input: {e}")
        except Exception as e: self.log_message(f"Prediction error: {e}\n{traceback.format_exc()}", is_error=True); self.results_text.append(f"[ERROR] Prediction failed: {e}"); QMessageBox.critical(self, "Prediction Error", f"Error: {e}")


    # --- Window Closing ---
    def closeEvent(self, event): print("Close triggered."); reply = QMessageBox.question(self, 'Exit', "Exit AndreiNET?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No); event.accept() if reply == QMessageBox.Yes else event.ignore()

# ==================================
#       Entry Point
# ==================================
def run_gui():
    if utils.MinMaxScaler is None or utils.train_test_split is None: app_check = QApplication.instance() or QApplication(sys.argv); QMessageBox.warning(None, "Dependency Missing", "Scikit-learn missing. Data split/norm disabled.\nInstall: pip install scikit-learn")
    try: QApplication.setStyle("Fusion")
    except Exception: pass
    app = QApplication.instance() or QApplication(sys.argv); main_window = AndreiNetApp(); main_window.show(); sys.exit(app.exec_())

if __name__ == '__main__':
    os.makedirs("datasets", exist_ok=True); os.makedirs("models", exist_ok=True); run_gui()