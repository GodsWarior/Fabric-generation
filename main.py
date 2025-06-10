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
import asyncio
from functools import partial
from transformers import *


class GenerationThread(QThread):
    progress_updated = pyqtSignal(int, str)
    generation_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, pipe, input_image_path, output_dir_path, num_generations, device):
        super().__init__()
        self.pipe = pipe
        self.input_image_path = input_image_path
        self.output_dir_path = output_dir_path
        self.num_generations = num_generations
        self.device = device
        self._is_running = True
        self.current_step = 0

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            asyncio.run(self.generate_patterns())
        except Exception as e:
            self.error_occurred.emit(str(e))

    async def generate_patterns(self):
        try:
            # Создаем папку для результатов
            self.output_dir_path.mkdir(exist_ok=True, parents=True)

            # Анализ изображения
            self.progress_updated.emit(0, "Анализ исходного изображения...")
            design = self.analyze_design(self.input_image_path)
            if not design:
                return

            self.progress_updated.emit(0,
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

            # Генерация вариантов
            total_steps = 18 * self.num_generations  # 18 шага на каждое изображение
            self.current_step = 0

            for i in range(1, self.num_generations + 1):
                if not self._is_running:
                    self.progress_updated.emit(0, "Генерация отменена")
                    return

                self.progress_updated.emit(
                    int(self.current_step / total_steps * 100),
                    f"Генерация варианта {i} из {self.num_generations}..."
                )

                # Создаем callback для отслеживания прогресса
                def update_progress(step, timestep, latents):
                    self.current_step = (i - 1) * 18 + step + 1
                    progress = int(self.current_step / total_steps * 100)
                    self.progress_updated.emit(
                        progress,
                        f"Генерация варианта {i} из {self.num_generations}... (шаг {step+1}/18)"
                    )
                    return not self._is_running  # Возвращаем True если нужно прервать

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    partial(
                        self.pipe,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init_image,
                        strength=0.55,
                        guidance_scale=9.0,
                        num_inference_steps=33,  # Уменьшаем для более плавного прогресса
                        generator=torch.Generator(self.device).manual_seed(42 + i),
                        eta=0.9,
                        callback=update_progress,
                        callback_steps=1
                    )
                )

                if not self._is_running:
                    self.progress_updated.emit(0, "Генерация отменена")
                    return

                result = result.images[0]

                if design['style'] == 'geometric':
                    result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
                else:
                    result = result.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.DETAIL)

                output_path = self.output_dir_path / f"pattern_v{i}.png"
                result.save(output_path, quality=95, subsampling=0)
                self.progress_updated.emit(
                    int((i * 33) / total_steps * 100),
                    f"Сохранен: {output_path.name}"
                )

            # Гарантированно завершаем прогресс на 100%
            self.progress_updated.emit(100, "Генерация завершена")

        except Exception as e:
            self.error_occurred.emit(str(e))

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
            self.error_occurred.emit(f"Ошибка анализа: {str(e)}")
            return None


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
        self.generation_thread = None

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
        self.generate_btn = self.create_button("✨ Сгенерировать паттерны", self.generate_pattern, is_primary=True)
        self.generate_btn.setFixedHeight(45)

        # Кнопка отмены
        self.cancel_btn = self.create_button("❌ Отменить генерацию", self.cancel_generation)
        self.cancel_btn.setFixedHeight(45)
        self.cancel_btn.setVisible(False)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 5px;
                text-align: center;
                background: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)


        # Статус бар
        self.status_bar = QLabel()
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("color: #aaaaaa; font-style: italic;")

        # Компоновка
        layout.addWidget(title)
        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addStretch()
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.progress_bar)
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

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.update_status(message)
        QApplication.processEvents()

    def generate_pattern(self):
        """Запуск асинхронной генерации паттернов"""
        if not self.input_image_path or not self.output_dir_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите изображение и папку для сохранения!")
            return

        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.warning(self, "Ошибка", "Генерация уже выполняется!")
            return

        self.generate_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setValue(0)

        self.generation_thread = GenerationThread(
            self.pipe,
            self.input_image_path,
            self.output_dir_path,
            self.gen_count.value(),
            self.device
        )
        self.generation_thread.progress_updated.connect(self.update_progress)
        self.generation_thread.generation_finished.connect(self.on_generation_finished)
        self.generation_thread.error_occurred.connect(self.on_generation_error)
        self.generation_thread.start()

    def cancel_generation(self):
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.stop()
            self.update_status("Генерация отменяется...", "warning")

    def on_generation_finished(self):
        self.generate_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        QMessageBox.information(self, "Готово", f"Успешно сгенерировано {self.gen_count.value()} паттернов!")
        self.update_status("Генерация завершена", "success")

    def on_generation_error(self, error_message):
        self.generate_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_message}")
        self.update_status(f"Ошибка: {error_message}", "error")

    def closeEvent(self, event):
        if self.generation_thread and self.generation_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Генерация выполняется',
                'Генерация еще выполняется. Вы уверены, что хотите закрыть приложение?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.generation_thread.stop()
                self.generation_thread.wait(2000)  # Даем время на завершение
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PatternGeneratorApp()
    window.show()
    sys.exit(app.exec_())