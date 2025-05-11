# python_gui/app.py
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import os
import traceback # Import for detailed error logging
import sys # For sys.stdout used in console progress bar

# Try to import the C++ extension
try:
    import andreinet_py as anp
except ImportError:
    messagebox.showerror("Import Error", "Could not import the C++ module 'andreinet_py'. "
                                         "Make sure it's built correctly and in the Python path.")
    anp = None
    # import sys
    # sys.exit(1) # Exit if C++ module is critical

# Activation Function Mapping
ACTIVATION_MAP_PY_TO_CPP = {"Linear": -1, "ReLU": 0, "Sigmoid": 1, "Softplus": 2}
ACTIVATION_MAP_CPP_TO_PY = {v: k for k, v in ACTIVATION_MAP_PY_TO_CPP.items()}


class AndreiNetApp(ctk.CTk):
    def __init__(self):
        super().__init__()
       
        self.total_epochs_for_progress_bar = 0 

        self.title("AndreiNet GUI")
        self.geometry("1250x850")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1) 
        self.grid_columnconfigure(1, weight=2) 
        self.grid_rowconfigure(0, weight=1)

        self.net_instance = None
        self.training_data_py = []
        self.current_file_path = ""
        self.training_thread = None
        self.stop_training_flag = threading.Event()
        self.gui_queue = queue.Queue()
        self.activation_combos = []

        # --- Left Panel (Controls) ---
        self.left_panel = ctk.CTkScrollableFrame(self, width=450)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- Data Loading Section ---
        data_frame = ctk.CTkFrame(self.left_panel)
        data_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(data_frame, text="1. Data Loading", font=("Arial", 16, "bold")).pack(pady=5, anchor="w")
        self.load_button = ctk.CTkButton(data_frame, text="Load CSV Data", command=self.load_csv)
        self.load_button.pack(pady=5, fill="x")
        self.file_label = ctk.CTkLabel(data_frame, text="No file loaded.")
        self.file_label.pack(pady=5)
        ctk.CTkLabel(data_frame, text="Target Column Index (0-indexed, -1 for last):").pack(anchor="w")
        self.target_col_entry = ctk.CTkEntry(data_frame, placeholder_text="-1")
        self.target_col_entry.insert(0, "-1")
        self.target_col_entry.pack(fill="x", pady=(0,5))

        # --- Network Structure Definition Section ---
        self.structure_frame = ctk.CTkFrame(self.left_panel)
        self.structure_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.structure_frame, text="2. Network Structure", font=("Arial", 16, "bold")).pack(pady=5, anchor="w")
        ctk.CTkLabel(self.structure_frame, text="Layer Sizes (comma-separated):").pack(anchor="w")
        self.layer_sizes_entry = ctk.CTkEntry(self.structure_frame, placeholder_text="e.g., num_features,h1,num_outputs")
        self.layer_sizes_entry.pack(fill="x", pady=(0,5))
        self.define_structure_button = ctk.CTkButton(self.structure_frame, text="Define/Update Network Structure", command=self.define_or_update_network_structure)
        self.define_structure_button.pack(pady=10, fill="x")

        # --- Network Properties (Activations, etc.) Section ---
        self.properties_frame = ctk.CTkFrame(self.left_panel)
        self.properties_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.properties_frame, text="3. Network Properties", font=("Arial", 16, "bold")).pack(pady=5, anchor="w")
        self.activations_label = ctk.CTkLabel(self.properties_frame, text="Layer Activations:")
        self.activations_label.pack(anchor="w", pady=(5,0))
        self.layer_activations_ui_container = ctk.CTkFrame(self.properties_frame)
        self.layer_activations_ui_container.pack(fill="x", pady=2)
        self.apply_properties_button = ctk.CTkButton(self.properties_frame, text="Apply Activation Changes", command=self.apply_activation_changes)
        self.apply_properties_button.pack(pady=10, fill="x")

        # --- Training Parameters Section ---
        train_params_frame = ctk.CTkFrame(self.left_panel)
        train_params_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(train_params_frame, text="4. Training", font=("Arial", 16, "bold")).pack(pady=5, anchor="w")
        ctk.CTkLabel(train_params_frame, text="Epochs:").pack(anchor="w")
        self.epochs_entry = ctk.CTkEntry(train_params_frame, placeholder_text="100")
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.pack(fill="x", pady=(0,5))
        ctk.CTkLabel(train_params_frame, text="Learning Rate:").pack(anchor="w")
        self.lr_entry = ctk.CTkEntry(train_params_frame, placeholder_text="0.01")
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.pack(fill="x", pady=(0,5))
        self.start_train_button = ctk.CTkButton(train_params_frame, text="Start Training", command=self.start_training_thread)
        self.start_train_button.pack(pady=5, side="left", expand=True, padx=(0,5))
        self.stop_train_button = ctk.CTkButton(train_params_frame, text="Stop Training", command=self.stop_training, fg_color="red")
        self.stop_train_button.pack(pady=5, side="right", expand=True, padx=(5,0))

        # --- Status and Utilities Section ---
        status_frame = ctk.CTkFrame(self.left_panel) # Renamed from status_frame to avoid conflict if used elsewhere
        status_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(status_frame, text="Utilities", font=("Arial", 16, "bold")).pack(pady=5, anchor="w")
        # self.accuracy_label = ctk.CTkLabel(status_frame, text="Accuracy (Train): N/A") # REMOVED
        # self.accuracy_label.pack(pady=5, anchor="w") # REMOVED
        self.reset_button = ctk.CTkButton(status_frame, text="Reset All", command=self.reset_application_state, fg_color="orange")
        self.reset_button.pack(pady=10, fill="x")

        # --- Right Panel (Visualizations & Log) ---
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_panel.grid_rowconfigure(0, weight=1); self.right_panel.grid_rowconfigure(1, weight=1); self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        net_viz_frame = ctk.CTkFrame(self.right_panel); net_viz_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(net_viz_frame, text="Network Structure", font=("Arial", 14)).pack()
        self.network_canvas = tk.Canvas(net_viz_frame, bg="white"); self.network_canvas.pack(fill="both", expand=True)
        plot_frame = ctk.CTkFrame(self.right_panel); plot_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.fig, (self.ax_data, self.ax_loss) = plt.subplots(1, 2, figsize=(8, 3)); self.fig.tight_layout(pad=2.0)
        self.ax_data.set_title("Data (Feature 0 vs Target)"); self.ax_loss.set_title("Training Loss (SSR)")
        self.ax_loss.set_xlabel("Epoch"); self.ax_loss.set_ylabel("Loss"); 
        self.loss_line, = self.ax_loss.plot([], [], 'r-', animated=False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame); self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True); self.canvas.draw()
        log_frame = ctk.CTkFrame(self.right_panel); log_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(log_frame, text="Log Console", font=("Arial", 14)).pack()
        self.log_console = ctk.CTkTextbox(log_frame, wrap="word", height=150); self.log_console.pack(fill="both", expand=True, pady=5)
        self.log_console.configure(state="disabled")

        self.protocol("WM_DELETE_WINDOW", self.on_closing_direct_exit) # MODIFIED
        self.update_gui_state()
        self.after(100, self.process_gui_queue)

    def on_closing_direct_exit(self): # RENAMED and MODIFIED
        # Attempt to stop training thread if it's running
        if self.training_thread and self.training_thread.is_alive():
            self.log_message("Window closed: Attempting to stop training thread...")
            self.stop_training_flag.set()
            self.training_thread.join(timeout=0.2) # Short timeout to allow graceful exit
        
        self.log_message("Exiting application...") # Optional: log before destroying
        self.destroy() # This cleanly closes the Tkinter window and ends mainloop

    def reset_application_state(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all data, network, and logs?"):
            self.log_message("--- Resetting Application State ---", clear_first=True)
            self.training_data_py = []
            self.current_file_path = ""
            self.file_label.configure(text="No file loaded.")
            self.target_col_entry.delete(0, tk.END); self.target_col_entry.insert(0, "-1")
            self.net_instance = None
            self.layer_sizes_entry.delete(0, tk.END)
            for widget in self.layer_activations_ui_container.winfo_children(): widget.destroy()
            self.activation_combos.clear()
            self.draw_network_visualization()
            self.epochs_entry.delete(0, tk.END); self.epochs_entry.insert(0, "100")
            self.lr_entry.delete(0, tk.END); self.lr_entry.insert(0, "0.01")
            
            self.ax_data.clear()
            self.ax_data.set_title("Data (Feature 0 vs Target)")
            self.ax_data.set_xlabel("Feature 0 (from remaining features)")
            self.ax_data.set_ylabel("Target")
            
            self.loss_epochs = []
            self.loss_values_ssr = []
            self.ax_loss.clear() 
            self.ax_loss.set_title("Training Loss (SSR)")
            self.ax_loss.set_xlabel("Epoch")
            self.ax_loss.set_ylabel("Loss")
            self.loss_line, = self.ax_loss.plot([], [], 'r-', animated=False)
            
            self.canvas.draw()

            # self.accuracy_label.configure(text="Accuracy (Train): N/A") # REMOVED
            # self.current_accuracy = 0.0 # REMOVED

            if self.training_thread and self.training_thread.is_alive():
                self.stop_training_flag.set()
                self.training_thread.join(timeout=0.2) 
            self.training_thread = None
            self.stop_training_flag.clear()
            self.update_gui_state()
            self.log_message("Application state has been reset.")

    def log_message(self, message, clear_first=False):
        self.log_console.configure(state="normal")
        if clear_first: self.log_console.delete("1.0", tk.END)
        self.log_console.insert(tk.END, message + "\n")
        self.log_console.see(tk.END) 
        self.log_console.configure(state="disabled")

    def load_csv(self):
        file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path: return
        try:
            df = pd.read_csv(file_path, header=None, dtype=float)
            self.log_message(f"Loaded {os.path.basename(file_path)} (Processed as NO HEADER, forced float). Shape: {df.shape}.")

            self.current_file_path = file_path
            self.file_label.configure(text=os.path.basename(file_path))
            if df.empty: self.log_message("Error: Loaded CSV is empty or unparsable."); return
            
            target_col_idx = int(self.target_col_entry.get())
            num_df_columns = df.shape[1]
            if num_df_columns == 0: self.log_message("Error: CSV has zero columns."); return
            if not (-num_df_columns <= target_col_idx < num_df_columns):
                self.log_message(f"Error: Target index {target_col_idx} out of bounds for {num_df_columns} columns."); return

            y_series = df.iloc[:, target_col_idx]
            column_to_drop_idx = df.columns[target_col_idx]
            X_df = df.drop(columns=[column_to_drop_idx])

            y_list_of_floats = [float(val) for val in y_series.values]
            X_list_of_list_of_floats = [([float(val) for val in X_df.iloc[r,:].values] if X_df.shape[1]>0 else []) for r in range(X_df.shape[0])]
            if X_df.shape[1] == 0: self.log_message("Warning: No feature columns after target removal.")
                
            self.training_data_py = list(zip(X_list_of_list_of_floats, [[fval] for fval in y_list_of_floats]))

            self.log_message(f"Processed {len(self.training_data_py)} samples into Python float lists.")
            if self.training_data_py:
                num_features = len(self.training_data_py[0][0])
                num_outputs = len(self.training_data_py[0][1])
                self.log_message(f"Detected: {num_features} features, {num_outputs} output(s).")
                if not self.layer_sizes_entry.get():
                    self.layer_sizes_entry.delete(0, tk.END)
                    default_input_nodes = num_features if num_features > 0 else 1
                    hidden_nodes = max(1, default_input_nodes // 2) if default_input_nodes > 1 else 1
                    if default_input_nodes == 1 and num_outputs == 1: hidden_nodes = 1
                    elif default_input_nodes > 1 and num_features >=2 : hidden_nodes = max(1, num_features //2 + num_features % 2 )
                    else: hidden_nodes = max(1, default_input_nodes // 2)
                    self.layer_sizes_entry.insert(0, f"{default_input_nodes},{hidden_nodes},{num_outputs}")
            
            self.ax_data.clear()
            if X_df.shape[1] > 0 and len(y_list_of_floats) > 0:
                self.ax_data.scatter(X_df.iloc[:, 0].astype(float), y_list_of_floats, alpha=0.5)
            elif len(y_list_of_floats) > 0 : 
                self.log_message("No features to plot for data viz (X_df has 0 columns).")
            self.ax_data.set_title("Data (Feature 0 vs Target)"); self.ax_data.set_xlabel("Feature 0"); self.ax_data.set_ylabel("Target")
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))
            self.log_message(f"General Error loading CSV: {type(e).__name__} - {e}\n{traceback.format_exc()}")

    def define_or_update_network_structure(self):
        if anp is None: self.log_message("C++ module not loaded."); return
        try:
            layer_sizes_str = self.layer_sizes_entry.get()
            if not layer_sizes_str: self.log_message("Error: Layer sizes cannot be empty."); return
            raw_sizes = [s.strip() for s in layer_sizes_str.split(',')]; layer_sizes = [int(s) for s in raw_sizes]
            if not all(s.isdigit() and int(s) > 0 for s in raw_sizes) or not layer_sizes :
                 self.log_message("Error: Layer sizes invalid."); return
            if self.net_instance: self.log_message("Existing network will be replaced.")
            self.net_instance = anp.Net(layer_sizes)
            self.log_message(f"New network created: {layer_sizes}")
            self.net_instance.randomize_all_weights(-0.5, 0.5); self.net_instance.set_all_biases(0.01)
            self.log_message("Initialized weights/biases.")
            self.rebuild_layer_activation_ui()
            self.apply_activation_changes(log_success=False)
            self.log_message(f"Network Structure:\n{self.net_instance.get_network_structure_string()}")
            self.draw_network_visualization()
        except Exception as e:
            messagebox.showerror("Network Structure Error", str(e))
            self.log_message(f"Network structure error: {e}\n{traceback.format_exc()}")
            self.net_instance = None
        finally: self.update_gui_state()

    def rebuild_layer_activation_ui(self):
        for widget in self.layer_activations_ui_container.winfo_children(): widget.destroy()
        self.activation_combos.clear()
        if not self.net_instance: self.update_gui_state(); return
        cpp_layers = self.net_instance.get_layers(); activation_options = list(ACTIVATION_MAP_PY_TO_CPP.keys())
        for i, layer_obj in enumerate(cpp_layers):
            frame = ctk.CTkFrame(self.layer_activations_ui_container); frame.pack(fill="x", pady=1, padx=1)
            label_text = f"L{i} ({layer_obj.node_count}N){' (In)' if i==0 else (' (Out)' if i==len(cpp_layers)-1 else ' (Hid)')}"
            ctk.CTkLabel(frame, text=label_text, width=100, anchor="w").pack(side="left", padx=(0,5))
            combo = ctk.CTkComboBox(frame, values=activation_options, width=120)
            current_py_act_name = ACTIVATION_MAP_CPP_TO_PY.get(layer_obj.nodes[0].activation_type if layer_obj.nodes else -1, "Linear")
            combo.set(current_py_act_name)
            combo.pack(side="left", padx=5, expand=True, fill="x"); self.activation_combos.append(combo)
        self.update_gui_state()

    def apply_activation_changes(self, log_success=True):
        if not self.net_instance or anp is None: self.log_message("Network/Module not ready."); return
        cpp_layers = self.net_instance.get_layers()
        if len(self.activation_combos) != len(cpp_layers): self.log_message("UI/Net layer mismatch."); return
        changed_any = False
        for i, combo in enumerate(self.activation_combos):
            selected_name = combo.get(); cpp_id = ACTIVATION_MAP_PY_TO_CPP.get(selected_name)
            current_cpp_id = cpp_layers[i].nodes[0].activation_type if cpp_layers[i].nodes else -1
            if cpp_id is not None and cpp_id != current_cpp_id:
                self.net_instance.set_layer_activation(i, cpp_id)
                if log_success: self.log_message(f"Applied: L{i} act set to {selected_name}.")
                changed_any = True
            elif cpp_id is None and log_success: self.log_message(f"Warn: Unknown act '{selected_name}' for L{i}.")
        if changed_any:
            if log_success: self.log_message("Activation changes applied.")
            self.log_message(f"Updated Network Structure:\n{self.net_instance.get_network_structure_string()}")
            self.draw_network_visualization()
        elif log_success: self.log_message("No activation changes detected.")
        self.update_gui_state()

    def update_gui_state(self):
        net_exists = self.net_instance is not None
        training_active = self.training_thread is not None and self.training_thread.is_alive()
        self.activations_label.configure(state=tk.NORMAL if net_exists else tk.DISABLED)
        self.apply_properties_button.configure(state=tk.NORMAL if net_exists and not training_active else tk.DISABLED)
        for combo in self.activation_combos: combo.configure(state=tk.NORMAL if net_exists and not training_active else tk.DISABLED)
        self.start_train_button.configure(state=tk.NORMAL if net_exists and self.training_data_py and not training_active else tk.DISABLED)
        self.stop_train_button.configure(state=tk.NORMAL if training_active else tk.DISABLED)
        self.define_structure_button.configure(state=tk.NORMAL if not training_active else tk.DISABLED)
        self.layer_sizes_entry.configure(state=tk.NORMAL if not training_active else tk.DISABLED)
        self.reset_button.configure(state=tk.NORMAL if not training_active else tk.DISABLED)

    def draw_network_visualization(self):
        self.network_canvas.delete("all")
        if not self.net_instance: self.update_gui_state(); return
        layers_cpp = self.net_instance.get_layers();
        if not layers_cpp: self.update_gui_state(); return
        canvas_width = self.network_canvas.winfo_width(); canvas_height = self.network_canvas.winfo_height()
        if canvas_width <=1 or canvas_height <=1: canvas_width = 400; canvas_height = 300; self.network_canvas.config(width=canvas_width, height=canvas_height)
        node_r = 8; layer_pad = 40; node_pad_v = 5
        layer_spacing = (canvas_width - 2 * layer_pad) / max(1, len(layers_cpp) -1 ) if len(layers_cpp) > 1 else canvas_width / 2
        max_nodes = max(l.node_count for l in layers_cpp) if layers_cpp else 0
        node_positions = []
        for i, l_obj in enumerate(layers_cpp):
            num_n = l_obj.node_count; x = layer_pad + i * layer_spacing; layer_pos = []
            total_h = (num_n - 1) * (2 * node_r + node_pad_v) if num_n > 1 else 0
            start_y = (canvas_height - total_h) / 2; start_y = max(start_y, node_r + node_pad_v + 10)
            for j in range(num_n):
                y = start_y + j * (2 * node_r + node_pad_v)
                layer_pos.append((x,y))
                clr = "lightblue"; act_s = ACTIVATION_MAP_CPP_TO_PY.get(l_obj.nodes[0].activation_type if l_obj.nodes else -1,"L")[0:3]
                if i==0: clr="lightgreen"; 
                elif i==len(layers_cpp)-1: clr="salmon"
                self.network_canvas.create_oval(x-node_r,y-node_r,x+node_r,y+node_r,fill=clr,outline="black")
                self.network_canvas.create_text(x,y-node_r-7,text=act_s,font=("Arial",7),anchor="s")
            node_positions.append(layer_pos)
        for i in range(len(layers_cpp)-1):
            for j_idx, (x1,y1) in enumerate(node_positions[i]):
                for k_idx, (x2,y2) in enumerate(node_positions[i+1]):
                    self.network_canvas.create_line(x1+node_r,y1,x2-node_r,y2,fill="gray",width=0.5)
        self.update_gui_state()

    def _print_console_progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        percent_val = iteration / float(total) if total > 0 else 0
        percent_str = ("{0:." + str(decimals) + "f}").format(100 * percent_val)
        filled_length = int(length * iteration // total) if total > 0 else 0
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent_str}% {suffix}')
        sys.stdout.flush()
        if iteration >= total : 
            sys.stdout.write('\n')
            sys.stdout.flush()

    def training_progress_callback(self, epoch, avg_ssr, avg_ce, time_ms, is_final):
        self.gui_queue.put(("progress", epoch, avg_ssr, avg_ce, time_ms, is_final))
        progress_suffix = f"Epoch {epoch}/{self.total_epochs_for_progress_bar}, SSR: {avg_ssr:.4f}"
        self._print_console_progress_bar(epoch, self.total_epochs_for_progress_bar, prefix='Training:', suffix=progress_suffix, length=40)
        # Accuracy calculation REMOVED from here

    def start_training_thread(self):
        if not self.net_instance: self.log_message("Error: Network not defined."); return
        if not self.training_data_py: self.log_message("Error: No training data."); return
        try: 
            epochs = int(self.epochs_entry.get()); lr = float(self.lr_entry.get())
            self.total_epochs_for_progress_bar = epochs
        except ValueError: self.log_message("Error: Invalid epochs/LR."); return
        
        self.stop_training_flag.clear()
        self.log_message(f"Starting training: {epochs} E, LR={lr}", clear_first=True)
        # self.accuracy_label.configure(text="Accuracy (Train): N/A") # REMOVED
        self.loss_epochs = []; self.loss_values_ssr = []
        self.loss_line.set_data([], [])
        self.ax_loss.relim(); self.ax_loss.autoscale_view() 
        self.canvas.draw()
        sys.stdout.write("\n") 
        self._print_console_progress_bar(0, self.total_epochs_for_progress_bar, prefix='Training:', suffix='Starting...', length=40)
        self.training_thread = threading.Thread(target=self.train_network_in_thread, args=(epochs, lr))
        self.training_thread.daemon = True; self.training_thread.start()
        self.update_gui_state()

    def train_network_in_thread(self, epochs, lr):
        self.log_message(f"[Thread] Starting C++ train. Samples: {len(self.training_data_py)}")
        try:
            self.net_instance.train(self.training_data_py, epochs, lr, 1, True, self.training_progress_callback)
            msg = "Training completed." if not self.stop_training_flag.is_set() else "Training stopped by user."
            self.gui_queue.put(("finished", msg))
            if not self.stop_training_flag.is_set() and epochs > 0:
                 # Use self.net_instance.average_ssr if available and populated by C++ train completion
                 last_ssr = self.net_instance.average_ssr if hasattr(self.net_instance, 'average_ssr') else 0.0
                 progress_suffix = f"Epoch {epochs}/{epochs}, SSR: {last_ssr:.4f}"
                 self._print_console_progress_bar(epochs, epochs, prefix='Training:', suffix=progress_suffix, length=40)
            else:
                sys.stdout.write("\nTraining Interrupted or 0 Epochs.\n"); sys.stdout.flush()
        except Exception as e:
            tb_str = traceback.format_exc()
            self.log_message(f"[Thread] EXCEPTION during C++ train: {type(e).__name__}: {e}\n{tb_str}")
            print(f"!!!!!!!!!! [Thread] EXCEPTION: {type(e).__name__}: {e}\n{tb_str}")
            self.gui_queue.put(("error", f"Training error: {str(e)}"))
            sys.stdout.write("\nTraining Error Occurred.\n"); sys.stdout.flush()
        finally:
            self.log_message("[Thread] C++ train call ended.")
            self.gui_queue.put(("enable_buttons", None))

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log_message("Attempting to stop training...")
            self.stop_training_flag.set() 
        self.update_gui_state()

    def process_gui_queue(self):
        try:
            while True:
                message_type, *payload = self.gui_queue.get_nowait()
                if message_type == "progress":
                    epoch, avg_ssr, avg_ce, time_ms, is_final = payload
                    self.log_message(f"E {epoch}: SSR={avg_ssr:.6f}, CE={avg_ce:.6f}, T={time_ms:.2f}ms")
                    self.loss_epochs.append(epoch); self.loss_values_ssr.append(avg_ssr)
                    self.loss_line.set_data(self.loss_epochs, self.loss_values_ssr)
                    self.ax_loss.relim(); self.ax_loss.autoscale_view(); self.canvas.draw()
                    if is_final: self.log_message("--- Final epoch C++ report ---")
                # REMOVED accuracy_update handling
                # elif message_type == "accuracy_update":
                #     self.accuracy_label.configure(text=f"Accuracy (Train): {payload[0]:.2f}%")
                elif message_type == "finished": self.log_message(payload[0])
                elif message_type == "error": self.log_message(f"Error: {payload[0]}")
                elif message_type == "enable_buttons":
                    self.training_thread = None
                    self.update_gui_state()
        except queue.Empty: pass
        except Exception as e:
            self.log_message(f"ERROR IN PROCESS_GUI_QUEUE: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally: self.after(100, self.process_gui_queue)

if __name__ == "__main__":
    app = AndreiNetApp()
    app.mainloop()