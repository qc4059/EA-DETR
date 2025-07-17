from login_widget import LoginWidget
import hashlib
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox, QFileDialog, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt  # 新增导入，用于 aspectRatioMode
from ultralytics import YOLO
import cv2
import numpy as np

class denglu(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化用户数据
        self.init_users()
        # 主窗口实例初始化为 None
        self.main_widget = None
        # 初始化UI
        self.init_ui()
        self.center_window()

    def center_window(self):
        """将窗口居中显示"""
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)

    def init_users(self):
        # 示例用户数据，实际应用中应该使用数据库
        self.users = {
            "admin": self.hash_password("admin123"),
            "test": self.hash_password("test123")
        }

    def hash_password(self, password):
        """使用SHA-256对密码进行哈希"""
        return hashlib.sha256(password.encode()).hexdigest()

    def init_ui(self):
        self.setWindowTitle('YOLO钢筋检测系统')

        # 创建堆叠窗口部件
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 创建登录窗口
        self.login_widget = LoginWidget()
        self.login_widget.login_successful.connect(self.on_login)
        self.login_widget.signup_requested.connect(self.on_signup)

        # 添加登录窗口到堆叠窗口部件
        self.stacked_widget.addWidget(self.login_widget)

        # 设置登录界面时的窗口大小
        self.resize(400, 300)

        # 设置初始显示登录界面
        self.stacked_widget.setCurrentIndex(0)

    def on_login(self, username, password):
        """处理登录请求"""
        # 计算密码哈希
        hashed_password = self.hash_password(password)

        if username in self.users and self.users[username] == hashed_password:
            try:
                if self.main_widget is None:
                    # 初始化YOLO模型
                    self.model = YOLO('D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.pt')
                    # 创建主界面
                    self.main_widget = QWidget()
                    layout = QVBoxLayout()

                    # 添加选择图像按钮
                    self.select_image_btn = QPushButton("选择图像进行检测")
                    self.select_image_btn.clicked.connect(self.select_image)
                    layout.addWidget(self.select_image_btn)

                    # 添加显示检测结果的标签
                    self.result_label = QLabel()
                    layout.addWidget(self.result_label)

                    self.main_widget.setLayout(layout)
                    self.stacked_widget.addWidget(self.main_widget)

                # 切换到主界面
                self.stacked_widget.setCurrentWidget(self.main_widget)

                # 获取主界面的大小并设置
                main_size = self.main_widget.size()
                self.resize(main_size)
                self.center_window()
                self.statusBar().showMessage(f"欢迎回来，{username}！")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载主界面时出错：{str(e)}")
                print(f"错误详情：{e}")
        else:
            QMessageBox.warning(self.login_widget, "错误", "用户名或密码错误！")

    def on_signup(self):
        """处理注册请求"""
        QMessageBox.information(self.login_widget, "提示", "请联系管理员创建账号")

    def closeEvent(self, event):
        """关闭窗口时的处理"""
        if self.main_widget is not None and hasattr(self.main_widget, 'stop_processing'):
            self.main_widget.stop_processing()
        event.accept()

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg)")
        if file_path:
            try:
                # 读取图像
                img = cv2.imread(file_path)
                # 进行检测
                results = self.model(img)
                # 获取检测结果图像
                result_img = results[0].plot()
                # 将 OpenCV 图像转换为 QPixmap
                height, width, channel = result_img.shape
                bytes_per_line = 3 * width
                q_img = QPixmap.fromImage(QImage(result_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped())
                # 在界面上显示检测结果
                self.result_label.setPixmap(q_img.scaled(self.result_label.width(), self.result_label.height(), aspectRatioMode=Qt.KeepAspectRatio))
            except Exception as e:
                QMessageBox.critical(self, "错误", f"检测过程中出错：{str(e)}")
                print(f"错误详情：{e}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = denglu()
    win.show()
    sys.exit(app.exec_())