import sys
import joblib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)

# Load model + features
data = joblib.load("enhanced_cart.pkl")
model = data["model"]
features = data["features"]

class FraudDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fraud Detection (Enhanced CART)")
        self.setGeometry(300, 200, 400, 400)

        layout = QVBoxLayout()

        # Dynamically create inputs based on saved feature list
        self.inputs = []
        for feature in features:
            label = QLabel(f"Enter {feature}:")
            layout.addWidget(label)

            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Numeric value for {feature}")
            self.inputs.append(line_edit)
            layout.addWidget(line_edit)

        self.predict_btn = QPushButton("Check Transaction")
        self.predict_btn.clicked.connect(self.predict_transaction)
        layout.addWidget(self.predict_btn)

        self.setLayout(layout)

    def predict_transaction(self):
        try:
            # Get user input in correct feature order
            values = [float(inp.text()) for inp in self.inputs]
            values = np.array(values).reshape(1, -1)

            # Predict fraud / legit
            result = model.predict(values)[0]

            if result == 1:
                QMessageBox.warning(self, "Result", "⚠️ Transaction is FRAUDULENT!")
            else:
                QMessageBox.information(self, "Result", "✅ Transaction is LEGITIMATE")

        except ValueError:
            QMessageBox.critical(self, "Error", "Please enter valid numeric values.")

# Run app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FraudDetectionApp()
    window.show()
    sys.exit(app.exec_())
