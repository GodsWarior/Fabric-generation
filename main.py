import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from colorthief import ColorThief
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

aaa = object

class PatternGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌀 Генератор текстильных паттернов")
        self.setGeometry(100, 100, 700, 450)
        self.setStyleSheet(self.dark_theme_stylesheet())

        # Переменные
        self.input_image_path = None
        self.output_dir_path = None
        self.num_generations = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # Инициализация UI
        self.init_ui()
        self.init_pipeline()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("🌀 <b>Генератор текстильных паттернов</b>")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; margin-bottom: 10px; color: #7bb5ff;")

        # Секция ввода
        input_group = self.create_group_box("Исходные данные")
        self.input_label = QLabel("Изображение не выбрано")
        self.input_label.setStyleSheet("color: #cccccc;")
        input_btn = self.create_button("📁 Выбрать изображение", self.select_input_image)

        input_layout = QVBoxLayout()
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(input_btn)
        input_group.setLayout(input_layout)

        # Секция вывода
        output_group = self.create_group_box("Сохранение результатов")
        self.output_label = QLabel("Папка не выбрана")
        self.output_label.setStyleSheet("color: #cccccc;")
        output_btn = self.create_button("💾 Выбрать папку", self.select_output_dir)

        # Настройки генерации
        self.gen_count = QSpinBox()
        self.gen_count.setRange(1, 10)
        self.gen_count.setValue(1)
        self.gen_count.setStyleSheet("background: #2d2d2d; color: white;")
        gen_count_layout = QHBoxLayout()
        gen_count_layout.addWidget(QLabel("Количество вариантов:"))
        gen_count_layout.addWidget(self.gen_count)
        gen_count_layout.addStretch()

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(output_btn)
        output_layout.addLayout(gen_count_layout)
        output_group.setLayout(output_layout)

        # Кнопка генерации
        generate_btn = self.create_button("✨ Сгенерировать паттерны", self.generate_pattern, is_primary=True)
        generate_btn.setFixedHeight(45)

        # Статус бар
        self.status_bar = QLabel()
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("color: #aaaaaa; font-style: italic;")

        # Компоновка
        layout.addWidget(title)
        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addStretch()
        layout.addWidget(generate_btn)
        layout.addWidget(self.status_bar)

        self.setCentralWidget(main_widget)

    def create_group_box(self, title):
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #7bb5ff;
            }
        """)
        return group

    def create_button(self, text, callback, is_primary=False):
        btn = QPushButton(text)
        btn.clicked.connect(callback)

        if is_primary:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border-radius: 6px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #5CBF60;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d2d;
                    color: #dcdcdc;
                    border: 1px solid #444;
                    border-radius: 5px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #3c3c3c;
                }
            """)

        return btn

    def dark_theme_stylesheet(self):
        return """
        QWidget {
            background-color: #1e1e1e;
            color: #dcdcdc;
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
        }
        QLabel {
            color: #cccccc;
        }
        QMessageBox {
            background-color: #2b2b2b;
        }
        QMessageBox QLabel {
            color: white;
        }
        QSpinBox {
            padding: 3px;
        }
        """

    def init_pipeline(self):
        """Инициализация модели Stable Diffusion"""
        try:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    subfolder="scheduler"
                )
            ).to(self.device)

            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass

            self.pipe.enable_attention_slicing()
            self.update_status("Модель загружена и готова к работе")
        except Exception as e:
            self.update_status(f"Ошибка загрузки модели: {str(e)}", "error")

    def select_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите исходное изображение",
            "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.input_image_path = Path(file_path)
            self.input_label.setText(f"<b>Выбрано:</b> {self.input_image_path}")
            self.update_status("Изображение загружено")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if dir_path:
            self.output_dir_path = Path(dir_path)
            self.output_label.setText(f"<b>Выбрана папка:</b> {self.output_dir_path}")
            self.update_status("Папка для сохранения выбрана")

    def update_status(self, message, message_type="info"):
        colors = {
            "info": "#aaaaaa",
            "success": "#4CAF50",
            "error": "#f44336",
            "warning": "#ff9800"
        }
        color = colors.get(message_type, "#aaaaaa")
        self.status_bar.setStyleSheet(f"color: {color}; font-style: italic;")
        self.status_bar.setText(message)
        QApplication.processEvents()

    def analyze_design(self, img_path):
        """Анализ характеристик исходного изображения"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return {
                    'colors': ["#5A7D9A", "#C3B8A5"],
                    'style': "geometric",
                    'complexity': "medium",
                    'texture': "smooth"
                }

            # Анализ цветов
            color_thief = ColorThief(str(img_path))
            palette = color_thief.get_palette(color_count=6, quality=3)
            hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]

            # Анализ стиля
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.countNonZero(edges) / (gray.size / 100)
            style = "organic" if edge_density < 15 else "geometric"
            complexity = "high" if edge_density > 25 else "low" if edge_density < 10 else "medium"

            # Анализ текстуры
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            texture_score = np.std(cv2.Laplacian(blur, cv2.CV_64F))
            texture_type = "rough" if texture_score > 500 else "moderate" if texture_score > 300 else "smooth"

            return {
                'colors': hex_colors,
                'style': style,
                'complexity': complexity,
                'texture': texture_type
            }
        except Exception as e:
            self.update_status(f"Ошибка анализа: {str(e)}", "error")
            return None

    def generate_pattern(self):
        """Основная функция генерации паттернов"""
        if not self.input_image_path or not self.output_dir_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите изображение и папку для сохранения!")
            return

        try:
            # Создаем папку для результатов
            self.output_dir_path.mkdir(exist_ok=True, parents=True)

            # Анализ изображения
            self.update_status("Анализ исходного изображения...")
            design = self.analyze_design(self.input_image_path)
            if not design:
                return

            self.update_status(
                f"Стиль: {design['style']}, Текстура: {design['texture']}, Цвета: {', '.join(design['colors'][:3])}...")

            # Подготовка промптов
            prompt = f"""
Professional {design['style']} pattern design:
- Color palette: {", ".join(design['colors'])}
- {design['complexity']} complexity with natural variations
- Balanced composition
- Seamless tiling capability
- High-quality {design['texture']} texture
- 8K resolution details
- Harmonious color transitions
- Professional textile design
"""
            negative_prompt = """
mirrored, symmetric, duplicated, repetitive, 
blurry, low-res, artifacts, watermark, text, 
logo, cropped, deformed, disfigured, ugly, 
tiled, grid, chessboard, perfect symmetry
"""

            # Подготовка изображения
            init_image = Image.open(self.input_image_path).convert("RGB")
            target_size = 768 if self.device == "cuda" else 512
            init_image = init_image.resize((target_size, target_size), Image.LANCZOS)

            # Прогресс-бар
            total = self.gen_count.value()
            progress = QProgressDialog("Генерация паттернов...", "Отмена", 0, total, self)
            progress.setWindowTitle("Генерация")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.show()

            # Генерация вариантов
            for i in range(1, total + 1):
                if progress.wasCanceled():
                    self.update_status("Генерация отменена", "warning")
                    break

                progress.setValue(i)
                self.update_status(f"Генерация варианта {i} из {total}...")
                QApplication.processEvents()  # Обновляем интерфейс

                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=0.55,
                    guidance_scale=9.0,
                    num_inference_steps=60,
                    generator=torch.Generator(self.device).manual_seed(42 + i),
                    eta=0.9
                ).images[0]


                if design['style'] == 'geometric':
                    result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
                else:
                    result = result.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.DETAIL)

                output_path = self.output_dir_path / f"pattern_v{i}.png"
                result.save(output_path, quality=95, subsampling=0)
                self.update_status(f"Сохранен: {output_path.name}")
                QApplication.processEvents()

            progress.close()

            QMessageBox.information(self, "Готово", f"Успешно сгенерировано {i} паттернов!")
            self.update_status("Генерация завершена", "success")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{str(e)}")
            self.update_status(f"Ошибка: {str(e)}", "error")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Установка стиля Fusion для более современного вида
    app.setStyle('Fusion')

    window = PatternGeneratorApp()
    window.show()
    sys.exit(app.exec_())