import sys
import joblib
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDialog, QTableView, QHeaderView,
    QMessageBox, QFrame, QGraphicsDropShadowEffect, QAbstractItemView,
    QScrollArea
)
from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QDoubleValidator, QColor, QFont
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel

# --- DATABASE SETUP ---

def init_database():
    """Initializes the SQLite database and creates the transactions table."""
    db = QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName("transaction_history.db")

    if not db.open():
        print("Error: Could not open database connection.")
        return False

    query = QSqlQuery()
    query.exec_("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            old_balance_cust REAL,
            new_balance_cust REAL,
            old_balance_recp REAL,
            new_balance_recp REAL,
            result TEXT
        )
    """)
    return True

# --- UI HELPER ---

def apply_shadow(widget):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(20)
    shadow.setXOffset(0)
    shadow.setYOffset(4)
    shadow.setColor(QColor(0, 0, 0, 30))
    widget.setGraphicsEffect(shadow)

# --- CUSTOM DIALOGS ---

class ErrorDialog(QDialog):
    def __init__(self, error_message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Validation Error")
        self.setModal(True)
        self.setFixedSize(400, 180)
        self.setStyleSheet("background-color: white; border-radius: 12px;")
        
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        
        content_layout = QHBoxLayout()
        icon_label = QLabel("❌")
        icon_label.setStyleSheet("font-size: 24pt;")
        content_layout.addWidget(icon_label, alignment=Qt.AlignTop)

        self.detail_label = QLabel(error_message)
        self.detail_label.setStyleSheet("font-size: 11pt; color: #333;")
        self.detail_label.setWordWrap(True)
        content_layout.addWidget(self.detail_label, stretch=1)
        
        main_layout.addLayout(content_layout)
        self.close_button = QPushButton("OK")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; color: white; border-radius: 6px; 
                padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        self.close_button.clicked.connect(self.accept)
        main_layout.addWidget(self.close_button, alignment=Qt.AlignCenter)


class ResultDialog(QDialog):
    def __init__(self, is_fraudulent, confidence, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Result")
        self.setModal(True)
        self.setFixedSize(450, 240)
        self.setStyleSheet("background-color: white; border-radius: 12px;")

        main_layout = QVBoxLayout(self)
        if is_fraudulent:
            res_text, res_icon = "FRAUDULENT", "🚨"
            detail = f"Model prediction: This transaction is flagged for fraud.\nConfidence: {confidence:.2%}"
            color = "#ef4444"
        else:
            res_text, res_icon = "LEGITIMATE", "✅"
            detail = f"Model prediction: This transaction appears legitimate.\nConfidence: {confidence:.2%}"
            color = "#10b981"

        self.result_label = QLabel(f"{res_icon} {res_text}")
        self.result_label.setStyleSheet(f"font-size: 20pt; font-weight: bold; color: {color}; margin-top: 10px;")
        main_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)

        self.detail_label = QLabel(detail)
        self.detail_label.setStyleSheet("font-size: 11pt; color: #4b5563; margin-bottom: 10px;")
        self.detail_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.detail_label, alignment=Qt.AlignCenter)

        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #f3f4f6; color: #374151; border-radius: 6px; 
                padding: 10px 30px; font-weight: bold; border: 1px solid #d1d5db;
            }
            QPushButton:hover { background-color: #e5e7eb; }
        """)
        self.close_button.clicked.connect(self.accept)
        main_layout.addWidget(self.close_button, alignment=Qt.AlignCenter)


# --- MAIN WINDOW ---

class TransactionForm(QMainWindow):
    ml_model = None
    ml_features = None

    def __init__(self):
        super().__init__()
        self.load_model()
        
        self.setWindowTitle("Transaction Analysis System")
        self.setMinimumSize(1000, 850)
        
        # Outer Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("background-color: #f8fafc;")
        self.setCentralWidget(self.scroll_area)

        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(30)

        self.setup_form_ui()
        self.setup_table_ui()

    # --- LOAD MODEL ---
    def load_model(self):
        model_filename = 'enhanced_cart_model.joblib'
        try:
            data = joblib.load(model_filename)
            if isinstance(data, dict) and 'model' in data:
                TransactionForm.ml_model = data.get("model")
                TransactionForm.ml_features = data.get("features")
            elif hasattr(data, 'predict'):
                TransactionForm.ml_model = data
            print("INFO: Model initialized successfully.")
        except Exception as e:
            print("ERROR: Model is not loaded. Analysis will not be available.")
            TransactionForm.ml_model = None

    def setup_form_ui(self):
        self.form_card = QFrame()
        self.form_card.setStyleSheet("QFrame { background-color: white; border-radius: 12px; }")
        apply_shadow(self.form_card)
        
        form_layout = QVBoxLayout(self.form_card)
        form_layout.setContentsMargins(40, 40, 40, 40)
        form_layout.setSpacing(15)
        
        header_label = QLabel("Transaction Analysis")
        header_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #1e293b; margin-bottom: 5px;")
        form_layout.addWidget(header_label)

        sub_label = QLabel("Enter the transaction details to evaluate fraud risk.")
        sub_label.setStyleSheet("color: #64748b; font-size: 10pt; margin-bottom: 15px;")
        form_layout.addWidget(sub_label)

        self.fields = [
            ("Transaction Amount", "e.g. 500.00"),
            ("Customer Old Balance", "e.g. 1200.50"),
            ("Customer New Balance", "e.g. 700.50"),
            ("Recipient Old Balance", "e.g. 0.00"),
            ("Recipient New Balance", "e.g. 500.00"),
        ]

        self.line_edits = {}
        locale = QLocale(QLocale.English, QLocale.UnitedStates)
        val = QDoubleValidator(0.0, 999999999.99, 2)
        val.setLocale(locale)

        for label_text, placeholder in self.fields:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(20)
            
            lbl = QLabel(label_text)
            lbl.setFixedWidth(180)
            lbl.setStyleSheet("font-size: 11pt; color: #475569; font-weight: 500;")
            
            edit = QLineEdit()
            edit.setValidator(val)
            edit.setPlaceholderText(placeholder)
            edit.setFixedHeight(45)
            edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #d1d5db; border-radius: 6px;
                    padding-left: 12px; font-size: 11pt; background-color: #ffffff;
                }
                QLineEdit:focus { border: 2px solid #3b82f6; }
            """)
            
            row_layout.addWidget(lbl)
            row_layout.addWidget(edit)
            self.line_edits[label_text] = edit
            form_layout.addLayout(row_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(200, 10, 0, 0)
        btn_layout.setSpacing(15)
        
        self.submit_btn = QPushButton("Submit Analysis")
        self.submit_btn.setFixedHeight(45)
        self.submit_btn.setFixedWidth(180)
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; color: white; border-radius: 6px;
                font-size: 11pt; font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        self.submit_btn.clicked.connect(self.submit_transaction)
        
        self.clear_btn = QPushButton("Cancel")
        self.clear_btn.setFixedHeight(45)
        self.clear_btn.setFixedWidth(100)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent; color: #3b82f6;
                font-size: 11pt; font-weight: 500; border: none;
            }
            QPushButton:hover { color: #1d4ed8; text-decoration: underline; }
        """)
        self.clear_btn.clicked.connect(self.clear_fields)
        
        btn_layout.addWidget(self.submit_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        form_layout.addLayout(btn_layout)

        self.main_layout.addWidget(self.form_card)

    def setup_table_ui(self):
        self.table_container = QFrame()
        self.table_container.setStyleSheet("background-color: white; border-radius: 12px;")
        apply_shadow(self.table_container)
        
        table_layout = QVBoxLayout(self.table_container)
        table_layout.setContentsMargins(30, 30, 30, 30)

        table_header_layout = QHBoxLayout()
        table_label = QLabel("Transaction History")
        table_label.setStyleSheet("font-size: 13pt; font-weight: bold; color: #1e293b;")
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setFixedWidth(100)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #fee2e2; color: #dc2626;
                border: 1px solid #fecaca; border-radius: 6px;
                font-weight: bold; padding: 6px;
            }
            QPushButton:hover { background-color: #fecaca; }
        """)
        self.delete_btn.clicked.connect(self.delete_selected_transaction)
        
        table_header_layout.addWidget(table_label)
        table_header_layout.addStretch()
        table_header_layout.addWidget(self.delete_btn)
        table_layout.addLayout(table_header_layout)

        # SQL Table Setup
        self.model = QSqlTableModel()
        self.model.setTable("transactions")
        self.model.setSort(0, Qt.DescendingOrder)
        self.model.select()

        headers = ["ID", "Timestamp", "Amount", "Customer Old Bal", "Customer New Bal", "Recipient Old Bal", "Recipient New Bal", "Result"]
        for i, h in enumerate(headers):
            self.model.setHeaderData(i, Qt.Horizontal, h)

        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        
        self.table_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_view.setFixedHeight(500)
        
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers) 
        self.table_view.setShowGrid(False)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.table_view.setStyleSheet("""
            QTableView {
                border: 1px solid #e2e8f0; background-color: white;
                selection-background-color: #3b82f6; selection-color: #ffffff;
                font-size: 10pt; color: #334155; outline: none;
            }
            QHeaderView::section {
                background-color: #f8fafc; padding: 10px; border: none;
                border-bottom: 2px solid #e2e8f0; font-weight: bold; color: #475569;
            }
        """)
        
        table_layout.addWidget(self.table_view)
        self.main_layout.addWidget(self.table_container)

    def clear_fields(self):
        for edit in self.line_edits.values():
            edit.clear()

    def submit_transaction(self):
        if TransactionForm.ml_model is None:
            ErrorDialog("Error: Prediction model not loaded.", self).exec_()
            return

        data = {}
        for lbl, edit in self.line_edits.items():
            text = edit.text().strip()
            if not text:
                ErrorDialog(f"'{lbl}' is required.", self).exec_()
                return
            data[lbl] = float(text)

        is_fraud, probs = self.evaluate_fraud_with_probs(data)
        result_text = "FRAUDULENT" if is_fraud else "LEGITIMATE"
        confidence = probs[1] if is_fraud else probs[0]

        # --- EVALUATION REPORT ---
        print("\n" + "="*50)
        print("   FRAUD EVALUATION")
        print("="*50)
        print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        print("INPUT FEATURES:")
        for key, value in data.items():
            print(f"  > {key:25}: {value:>12.2f}")
        print("-" * 50)
        print("MODEL PROBABILITIES:")
        print(f"  [0] Confidence in Legitimate: {probs[0]:>10.2%}")
        print(f"  [1] Confidence in Fraudulent: {probs[1]:>10.2%}")
        print("-" * 50)
        print(f"FINAL DECISION : {result_text}")
        print(f"CONSENSUS SCORE: {confidence:.2%}")
        print("="*50 + "\n")

        self.save_to_db(data, result_text)
        ResultDialog(is_fraud, confidence, self).exec_()

    def evaluate_fraud_with_probs(self, data):
        """Uses AdaBoost predict_proba to show the weighted decision of the trees."""
        model = TransactionForm.ml_model
        try:
            input_values = [
                data["Transaction Amount"], 
                data["Customer Old Balance"],
                data["Customer New Balance"], 
                data["Recipient Old Balance"],
                data["Recipient New Balance"]
            ]
            
            # Reshape for single prediction: (1, 5)
            input_array = np.array([input_values])

            # predict_proba returns a list of probabilities [P(Legit), P(Fraud)]
            probabilities = model.predict_proba(input_array)[0]
            prediction = np.argmax(probabilities)
            
            return bool(prediction == 1), probabilities
        except Exception as e:
            print(f"ERROR: Model evaluation failed: {e}")
            return False, [1.0, 0.0]

    def save_to_db(self, data, result):
        query = QSqlQuery()
        query.prepare("""
            INSERT INTO transactions (
                timestamp, amount, old_balance_cust, new_balance_cust, 
                old_balance_recp, new_balance_recp, result
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """)
        query.addBindValue(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        query.addBindValue(data["Transaction Amount"])
        query.addBindValue(data["Customer Old Balance"])
        query.addBindValue(data["Customer New Balance"])
        query.addBindValue(data["Recipient Old Balance"])
        query.addBindValue(data["Recipient New Balance"])
        query.addBindValue(result)
        
        if query.exec_():
            self.model.select()
            self.table_view.scrollToTop()

    def delete_selected_transaction(self):
        selected_index = self.table_view.currentIndex()
        if not selected_index.isValid():
            QMessageBox.information(self, "Selection", "Select a record to delete.")
            return

        confirm = QMessageBox.question(self, "Confirm", "Delete this record?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.model.removeRow(selected_index.row())
            self.model.submitAll()
            self.model.select()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    if not init_database():
        sys.exit(-1)

    window = TransactionForm()
    window.showMaximized()
    sys.exit(app.exec_())