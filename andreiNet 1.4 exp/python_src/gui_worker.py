# python_src/gui_worker.py
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot
import time # For potential sleeps if needed for progress simulation

# Import C++ bindings - robustness check
try:
    import andreinet_bindings as anet
except ImportError:
    print("FATAL ERROR: gui_worker.py - Could not import 'andreinet_bindings'.")
    print("Ensure C++ bindings are compiled and accessible.")
    anet = None

class TrainingSignals(QObject):
    """ Defines signals available from the training worker thread. """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    # Progress signal (epoch / total_epochs) - Requires train to run 1 epoch at a time
    # progress = pyqtSignal(int)
    status_update = pyqtSignal(str)


class TrainingWorker(QRunnable):
    """ Worker thread for running the C++ network training function. """
    def __init__(self, net_instance, training_data, epochs, initial_lr, batch_size, shuffle, validation_data=None, early_stopping_patience=None):
        super().__init__()
        if anet is None:
            raise ImportError("AndreiNET bindings not loaded in TrainingWorker.")

        self.net = net_instance
        self.training_data_cpp = training_data # Assumes data is already formatted for C++
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Add validation data and early stopping params
        self.validation_data_cpp = validation_data
        self.early_stopping_patience = early_stopping_patience if early_stopping_patience is not None and early_stopping_patience > 0 else None

        self.signals = TrainingSignals()

    @pyqtSlot()
    def run(self):
        """Execute the training task."""
        # --- NOTE: This currently calls the C++ 'train' for ALL epochs at once. ---
        # --- For true epoch-by-epoch progress and early stopping based on C++ ---
        # --- calculations, the C++ `train` function would need modification  ---
        # --- to run only one epoch and return loss/metrics.               ---

        # --- Simple implementation: Run all epochs, report start/end ---
        try:
            start_time = time.time()
            self.signals.status_update.emit(f"Starting C++ training ({self.epochs} epochs)...")

            # Call the potentially long-running C++ train function
            self.net.train(
                self.training_data_cpp,
                self.epochs,
                self.initial_lr,
                self.batch_size,
                self.shuffle
            )
            end_time = time.time()
            duration = end_time - start_time
            self.signals.status_update.emit(f"C++ training finished in {duration:.2f}s.")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"ERROR during training thread:\n{error_msg}")
            self.signals.error.emit((type(e).__name__, str(e)))
        finally:
            self.signals.finished.emit()

        # --- Alternative (Conceptual) for epoch-by-epoch progress/early stopping ---
        # --- Requires C++ `train_one_epoch` binding & `calculate_loss` binding ---
        # if False: # Disabled for now
        #     try:
        #         best_val_loss = float('inf')
        #         epochs_no_improve = 0
        #         for epoch in range(self.epochs):
        #             # 1. Train one epoch (requires new C++ binding)
        #             # train_loss = self.net.train_one_epoch(...)
        #             # self.signals.status_update.emit(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f}")
        #
        #             # 2. Validation step (requires new C++ binding)
        #             # if self.validation_data_cpp:
        #             #     val_loss = self.net.calculate_dataset_loss(self.validation_data_cpp)
        #             #     self.signals.status_update.emit(f"Epoch {epoch+1}/{self.epochs} | Val Loss: {val_loss:.4f}")
        #             #     if self.early_stopping_patience:
        #             #         if val_loss < best_val_loss:
        #             #             best_val_loss = val_loss
        #             #             epochs_no_improve = 0
        #             #             # Optional: Save best model weights here
        #             #         else:
        #             #             epochs_no_improve += 1
        #             #             if epochs_no_improve >= self.early_stopping_patience:
        #             #                 self.signals.status_update.emit(f"Early stopping triggered at epoch {epoch+1}.")
        #             #                 break # Exit training loop
        #
        #             # 3. Emit Progress
        #             # progress_percent = int(((epoch + 1) / self.epochs) * 100)
        #             # self.signals.progress.emit(progress_percent)
        #
        #             # Add small sleep to allow GUI to update if needed
        #             # time.sleep(0.01)
        #
        #         self.signals.status_update.emit("Training finished.")
        #
        #     except Exception as e:
        #         # ... error handling ...
        #     finally:
        #         self.signals.finished.emit()