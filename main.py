import sys
import joblib 
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDialog
)
from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QDoubleValidator

# --- Helper Functions ---

def load_qss(filepath):
    """Loads the QSS file content."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Stylesheet file '{filepath}' not found.")
        return ""

# --- Custom Dialogs ---

class ErrorDialog(QDialog):
    """A custom dialog to show input errors."""
    def __init__(self, error_message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Validation Error")
        self.setModal(True)
        self.setFixedSize(400, 180)
        self.setObjectName("ErrorDialog") 

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)

        # Icon and Header/Message
        content_layout = QHBoxLayout()
        content_layout.setAlignment(Qt.AlignCenter)
        
        # Error Icon (Red X or equivalent warning)
        icon_label = QLabel("❌")
        icon_label.setObjectName("ErrorIcon")
        content_layout.addWidget(icon_label, alignment=Qt.AlignTop | Qt.AlignLeft)

        # Detail Message
        self.detail_label = QLabel(error_message)
        self.detail_label.setObjectName("ErrorDetailMessage")
        self.detail_label.setWordWrap(True)
        content_layout.addWidget(self.detail_label, stretch=1)
        
        main_layout.addLayout(content_layout)

        # Close Button
        self.close_button = QPushButton("OK")
        self.close_button.setObjectName("ErrorCloseButton")
        self.close_button.clicked.connect(self.accept)
        # Wrap button in layout to center it
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.close_button, alignment=Qt.AlignCenter)
        main_layout.addLayout(button_layout)


class ResultDialog(QDialog):
    """A custom dialog to show the transaction result (Legitimate/Fraudulent)."""
    def __init__(self, is_fraudulent, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transaction Analysis Result")
        self.setModal(True)
        self.setFixedSize(400, 200)

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)

        # Determine the message and styling
        if is_fraudulent:
            result_text = "FRAUDULENT"
            style_class = "Fraudulent"
            icon_emoji = "🚨"
            detail_message = "Model prediction: This transaction is flagged for fraud."
        else:
            result_text = "LEGITIMATE"
            style_class = "Legitimate"
            icon_emoji = "✅"
            detail_message = "Model prediction: This transaction appears legitimate."

        # Result Label 
        self.result_label = QLabel(f"{icon_emoji} {result_text}")
        self.result_label.setObjectName(style_class)
        main_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)

        # Detail Message
        self.detail_label = QLabel(detail_message)
        self.detail_label.setObjectName("DetailMessage")
        main_layout.addWidget(self.detail_label, alignment=Qt.AlignCenter)

        # Close Button
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("CloseButton")
        self.close_button.clicked.connect(self.accept)
        main_layout.addWidget(self.close_button, alignment=Qt.AlignCenter)


# --- Main Application Window ---

class TransactionForm(QMainWindow):
    """Main window for entering transaction details."""
    
    # Class attributes for the ML model and expected features
    ml_model = None
    ml_features = None

    def __init__(self):
        super().__init__()
        
        # Input fields mapping (Label Text: Placeholder Text)
        self.fields = {
            "Transaction Amount": ("0.00", "amount_line"),
            "Customer Old Balance": ("Previous balance (CUST)", "old_cust_line"),
            "Customer New Balance": ("Current balance (CUST)", "new_cust_line"),
            "Recipient Old Balance": ("Previous balance (RECP)", "old_recp_line"),
            "Recipient New Balance": ("Current balance (RECP)", "new_recp_line"),
        }

        # Load the model. 
        self.load_model()
        
        self.setWindowTitle("Financial Transaction Analyzer")
        self.setObjectName("MainWindow")
        self.setFixedSize(600, 550)

        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Set up a validator for numerical input
        locale = QLocale(QLocale.English, QLocale.UnitedStates)
        self.double_validator = QDoubleValidator()
        self.double_validator.setLocale(locale)
        self.double_validator.setRange(0.00, 999999999.99, 2)
        self.double_validator.setNotation(QDoubleValidator.StandardNotation)

        # Header
        header = QLabel("Enter Transaction Details")
        header.setObjectName("HeaderLabel")
        main_layout.addWidget(header, alignment=Qt.AlignCenter)

        # Form Layout
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(15)
        form_layout.setContentsMargins(50, 20, 50, 20)

        self.line_edits = {}

        for label_text, (placeholder, obj_name) in self.fields.items():
            self.line_edits[label_text] = self._create_input_field(
                label_text, placeholder, obj_name, self.double_validator
            )
            form_layout.addLayout(self.line_edits[label_text])

        main_layout.addWidget(form_widget)

        # Button Layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        button_layout.setAlignment(Qt.AlignCenter)

        self.submit_button = QPushButton("Submit")
        self.submit_button.setObjectName("SubmitButton")
        self.submit_button.clicked.connect(self.submit_transaction)
        button_layout.addWidget(self.submit_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("ClearButton")
        self.clear_button.clicked.connect(self.clear_fields)
        button_layout.addWidget(self.clear_button)

        main_layout.addLayout(button_layout)
        main_layout.addStretch(1) # Push content to the top
        
    def load_model(self):
        """
        Loads the pickled machine learning model ('enhanced_cart.pkl') using joblib.
        """
        model_filename = 'enhanced_cart.pkl'
        
        try:
            print(f"INFO: Attempting to load ML model ({model_filename}) using joblib...")
            
            # 1. Load the data dictionary from the file using joblib
            data = joblib.load(model_filename)
            
            # 2. Extract model and features
            model_candidate = data.get("model")
            features_candidate = data.get("features")

            if callable(getattr(model_candidate, 'predict', None)):
                TransactionForm.ml_model = model_candidate
                TransactionForm.ml_features = features_candidate 
                print(f"INFO: Model '{model_filename}' loaded successfully. Type: {type(model_candidate)}")
                if TransactionForm.ml_features:
                    print(f"INFO: Model expects {len(TransactionForm.ml_features)} features: {TransactionForm.ml_features}")
                else:
                    print("WARNING: Feature list ('features' key) not found in the loaded file. Assuming feature order is based on input field order.")
                
                # The warning check has been removed here.

            else:
                print(f"ERROR: Object loaded from PKL file does not have a 'predict' method or 'model' key is missing.")
                print(f"Loaded object type: {type(model_candidate)}. Expected a model object.")
                TransactionForm.ml_model = None
            
        except FileNotFoundError:
            print(f"ERROR: Model file not found: {model_filename}. Running in mock mode.")
            TransactionForm.ml_model = None
        except Exception as e:
            print(f"ERROR: Failed to load ML model from {model_filename}: {e}. Running in mock mode.")
            TransactionForm.ml_model = None


    def _create_input_field(self, label_text, placeholder, obj_name, validator):
        """Creates a label/line edit pair in an HBox."""
        h_layout = QHBoxLayout()

        label = QLabel(label_text)
        label.setObjectName("InputLabel")
        h_layout.addWidget(label, 40) # 40% width for label

        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setObjectName(obj_name)
        line_edit.setValidator(validator)
        h_layout.addWidget(line_edit, 60) # 60% width for input

        return h_layout

    def clear_fields(self):
        """Clears all input fields."""
        for layout in self.line_edits.values():
            line_edit = layout.itemAt(1).widget() # The QLineEdit is at index 1
            line_edit.clear()

    def get_numerical_data(self):
        """Extracts and converts input data to floats. Returns (data, error_message)."""
        data = {}
        for label_text, layout in self.line_edits.items():
            line_edit = layout.itemAt(1).widget()
            text = line_edit.text().replace(',', '') # Remove thousand separators if they sneak in

            if not text:
                return None, f"'{label_text}' cannot be empty."

            try:
                data[label_text] = float(text)
            except ValueError:
                return None, f"Invalid numerical value for '{label_text}'."

        return data, None

    def check_transaction(self, data):
        """
        Uses the loaded ML model to predict if the transaction is legitimate or fraudulent.
        
        Args:
            data (dict): Dictionary containing the numerical transaction inputs.
            
        Returns:
            bool: True if fraudulent (prediction is 1), False if legitimate (prediction is 0).
        """
        model = TransactionForm.ml_model
        features = TransactionForm.ml_features
        
        feature_map = {
            "Transaction Amount": "amount",
            "Customer Old Balance": "oldbalanceOrg",
            "Customer New Balance": "newbalanceOrig",
            "Recipient Old Balance": "oldbalanceDest",
            "Recipient New Balance": "newbalanceDest",
        }
        
        if model is None:
            # Fallback logic if the model could not be loaded
            warning_dialog = ErrorDialog(
                "ML Model not found (enhanced_cart.pkl). Running a basic check (Transaction Amount > $10000 is FRAUD).",
                self
            )
            warning_dialog.setWindowTitle("Model Warning")
            warning_dialog.exec_()
            
            # Simple mock logic for demonstration purposes when model fails to load
            return data.get("Transaction Amount", 0) > 10000.00
            
        try:
            # 1. Prepare data for the model inference
            if not features:
                # If feature list is missing, assume order based on the feature_map keys
                # This ensures the order of input matches the order the model was trained on 
                # (which is ['amount', 'oldbalanceOrg', ...])
                ordered_keys = list(feature_map.keys())
            else:
                # If features are present, use the model's expected feature names to pull data
                # We reverse map the model's feature names to the UI labels
                reverse_map = {v: k for k, v in feature_map.items()}
                ordered_keys = [reverse_map.get(f, f) for f in features]
            
            # Create a list of feature values in the correct order
            feature_values = [data[key] for key in ordered_keys]
            
            # Convert to a 2D NumPy array (1 sample, N features)
            input_array = np.array([feature_values])
            
            # 2. Call the model's predict method
            prediction = model.predict(input_array)
            
            # 3. Interpret the result
            # The model returns 1 for Fraudulent and 0 for Legitimate
            is_fraudulent = prediction[0] == 1 
            
            print(f"ML Model Prediction: {prediction[0]}. Result: {'FRAUDULENT' if is_fraudulent else 'LEGITIMATE'}")
            
            return is_fraudulent

        except Exception as e:
            # Replaced QMessageBox.critical with custom ErrorDialog
            error_msg = f"Model prediction failed: {e}. Defaulting to FRAUDULENT. (Check feature names/order in check_transaction.)"
            error_dialog = ErrorDialog(error_msg, self)
            error_dialog.setWindowTitle("Prediction Critical Error")
            error_dialog.exec_()
            return True # Default to fraudulent on prediction error


    def submit_transaction(self):
        """Handles the submission, validation, and result display."""
        data, error = self.get_numerical_data()

        if error:
            # Replaced QMessageBox.critical with custom ErrorDialog
            error_dialog = ErrorDialog(error, self)
            error_dialog.exec_()
            return

        # Perform the fraud check using the custom model function
        is_fraudulent = self.check_transaction(data)

        # Show the result pop-up window
        result_dialog = ResultDialog(is_fraudulent, self)
        result_dialog.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qss = load_qss("styles.qss")
    app.setStyleSheet(qss)

    window = TransactionForm()
    window.show()
    sys.exit(app.exec_())