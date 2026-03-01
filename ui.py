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
from PyQt5.QtCore import Qt, QLocale, QRectF, QPointF, QSize
from PyQt5.QtGui import QDoubleValidator, QColor, QFont, QPainter, QPen, QBrush
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

# --- TREE VISUALIZATION DIALOG ---

class VisualTreeDialog(QDialog):
    """Popup dialog to graphically display the decision tree structure with scrolling and zoom support."""
    def __init__(self, tree_data, features, input_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Decision Tree Logic Visualizer")
        self.resize(1100, 800)
        self.tree = tree_data
        self.features = features
        self.input_data = input_data
        self.zoom_factor = 1.0  
        self.setStyleSheet("background-color: #ffffff; border-radius: 8px;")
        
        main_layout = QVBoxLayout(self)
        
        header = QLabel("Decision Path Analysis")
        header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #1e293b; margin: 10px;")
        main_layout.addWidget(header, alignment=Qt.AlignCenter)

        # Scroll Area for the Tree
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False) 
        self.scroll_area.setStyleSheet("border: 1px solid #e2e8f0; background-color: #f8fafc;")
        
        # Calculate base dimensionS
        depth = self.get_max_depth(0)
        self.base_width = max(2400, (min(depth, 12)**2) * 120) 
        self.base_height = max(1200, depth * 180)

        self.canvas = QWidget()
        self.update_canvas_size()
        self.canvas.paintEvent = self.draw_tree_canvas
        
        # Enable mouse tracking for zoom
        self.canvas.setMouseTracking(True)
        self.canvas.wheelEvent = self.handle_wheel_zoom
        
        self.scroll_area.setWidget(self.canvas)
        main_layout.addWidget(self.scroll_area)

        # Legend
        footer = QLabel("Blue Nodes: Features | Green: Legit Leaf | Red: Fraud Leaf | Thick Blue Line: Path Taken | Ctrl+Scroll to Zoom")
        footer.setStyleSheet("color: #64748b; font-size: 10pt; margin: 10px;")
        main_layout.addWidget(footer, alignment=Qt.AlignCenter)

        close_btn = QPushButton("Close Visualizer")
        close_btn.setFixedWidth(150)
        close_btn.setFixedHeight(40)
        close_btn.setStyleSheet("""
            QPushButton { background-color: #3b82f6; color: white; border-radius: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #2563eb; }
        """)
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn, alignment=Qt.AlignCenter)

    def update_canvas_size(self):
        """Update canvas size based on zoom factor."""
        w = int(self.base_width * self.zoom_factor)
        h = int(self.base_height * self.zoom_factor)
        # Cap to prevent overflow
        w = min(w, 32000)
        h = min(h, 32000)
        self.canvas.setFixedSize(w, h)

    def handle_wheel_zoom(self, event):
        """Handle zooming when Ctrl is held while scrolling."""
        if event.modifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1
            
            # Constraints
            self.zoom_factor = max(0.2, min(self.zoom_factor, 3.0))
            
            self.update_canvas_size()
            self.canvas.update()
            event.accept()
        else:
            event.ignore()

    def get_max_depth(self, node_id):
        if self.tree.children_left[node_id] == -1:
            return 1
        return 1 + max(self.get_max_depth(self.tree.children_left[node_id]), 
                       self.get_max_depth(self.tree.children_right[node_id]))

    def draw_tree_canvas(self, event):
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Apply global scaling based on zoom
        painter.scale(self.zoom_factor, self.zoom_factor)
        
        if self.tree:
            start_x = self.base_width / 2
            start_y = 100
            initial_spread = self.base_width / 4.5
            self.draw_node(painter, 0, start_x, start_y, initial_spread, 1)

    def draw_node(self, painter, node_id, x, y, x_offset, depth):
        radius = 32
        v_gap = 150
        is_leaf = self.tree.children_left[node_id] == -1
        
        if not is_leaf:
            feat_idx = self.tree.feature[node_id]
            feat_name = self.features[feat_idx]
            val = self.input_data.get(feat_name, 0.0)
            threshold = self.tree.threshold[node_id]
            
            left_child = self.tree.children_left[node_id]
            right_child = self.tree.children_right[node_id]
            
            goes_left = (val <= threshold)

            # Draw Left Branch
            painter.setPen(QPen(QColor("#3b82f6" if goes_left else "#cbd5e1"), 5 if goes_left else 1.5))
            painter.drawLine(QPointF(x, y), QPointF(x - x_offset, y + v_gap))
            self.draw_node(painter, left_child, x - x_offset, y + v_gap, x_offset * 0.52, depth + 1)

            # Draw Right Branch
            painter.setPen(QPen(QColor("#3b82f6" if not goes_left else "#cbd5e1"), 5 if not goes_left else 1.5))
            painter.drawLine(QPointF(x, y), QPointF(x + x_offset, y + v_gap))
            self.draw_node(painter, right_child, x + x_offset, y + v_gap, x_offset * 0.52, depth + 1)

        # Draw the Node Circle
        if is_leaf:
            is_fraud = self.tree.value[node_id][0][1] > self.tree.value[node_id][0][0]
            painter.setBrush(QBrush(QColor("#ef4444" if is_fraud else "#10b981")))
        else:
            painter.setBrush(QBrush(QColor("#3b82f6")))
        
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(x, y), radius, radius)

        # Draw Text inside the Node
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        if is_leaf:
            label = "FRAUD" if self.tree.value[node_id][0][1] > self.tree.value[node_id][0][0] else "LEGIT"
            painter.drawText(QRectF(x-radius, y-radius, radius*2, radius*2), Qt.AlignCenter, label)
        else:
            feat_label = self.features[self.tree.feature[node_id]].replace(" ", "\n")
            painter.drawText(QRectF(x-radius, y-radius, radius*2, radius*2), Qt.AlignCenter, feat_label)
            
            painter.setPen(QPen(QColor("#1e293b")))
            painter.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
            painter.drawText(QPointF(x + radius + 8, y + 5), f"≤ {self.tree.threshold[node_id]:.1f}")

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
    ml_features = ["Amount", "Cust Old Bal", "Cust New Bal", "Recp Old Bal", "Recp New Bal"]

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
        model_filename = 'base_model.joblib'
        try:
            data = joblib.load(model_filename)
            if isinstance(data, dict) and 'model' in data:
                TransactionForm.ml_model = data.get("model")
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

        # --- FEATURE IMPORTANCE CALCULATION ---
        importances = TransactionForm.ml_model.feature_importances_
        feat_map = list(zip(self.ml_features, importances))
        feat_map.sort(key=lambda x: x[1], reverse=True)
        
        top_feature_name, top_feature_val = feat_map[0]

        # --- TERMINAL EVALUATION REPORT ---
        print("\n" + "="*60)
        print("    FRAUD EVALUATION")
        print("="*60)
        print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        print("INPUT FEATURES:")
        for key, value in data.items():
            print(f"  > {key:25}: {value:>15.2f}")
        print("-" * 60)
        print("FEATURE INFLUENCE:")
        print(f"  FEATURE: {top_feature_name}")
        print(f"  WEIGHT : {top_feature_val:.2%}")
        print("-" * 60)
        print("MODEL PROBABILITIES:")
        print(f"  Confidence in Legitimate: {probs[0]:>10.2%}")
        print(f"  Confidence in Fraudulent: {probs[1]:>10.2%}")
        print("-" * 60)
        print(f"FINAL DECISION : {result_text}")
        print(f"CONSENSUS SCORE: {confidence:.2%}")
        print("="*60 + "\n")

        input_mapped = {
            "Amount": data["Transaction Amount"],
            "Cust Old Bal": data["Customer Old Balance"],
            "Cust New Bal": data["Customer New Balance"],
            "Recp Old Bal": data["Recipient Old Balance"],
            "Recp New Bal": data["Recipient New Balance"]
        }

        self.save_to_db(data, result_text)
        ResultDialog(is_fraud, confidence, self).exec_()

        try:
            if hasattr(TransactionForm.ml_model, 'estimators_'):
                tree_data = TransactionForm.ml_model.estimators_[0].tree_
            else:
                tree_data = TransactionForm.ml_model.tree_
            VisualTreeDialog(tree_data, self.ml_features, input_mapped, self).exec_()
        except Exception as e:
            print(f"Visual Tree Popup Failed: {e}")

    def evaluate_fraud_with_probs(self, data):
        model = TransactionForm.ml_model
        try:
            input_values = [
                data["Transaction Amount"], 
                data["Customer Old Balance"],
                data["Customer New Balance"], 
                data["Recipient Old Balance"], 
                data["Recipient New Balance"]
            ]
            input_array = np.array([input_values])
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