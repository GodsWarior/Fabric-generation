import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from colorthief import ColorThief
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

# Конфигурация
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INPUT_IMAGE = Path(r"C:\fabric\img\1.jpg")
OUTPUT_DIR = Path(r"C:\fabric\results")
NUM_GENERATIONS = 1


def analyze_design(img_path):
    """Комплексный анализ узора с определением ключевых характеристик"""
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            'colors': ["#5A7D9A", "#C3B8A5"],
            'style': "geometric",
            'complexity': "medium",
            'symmetry': "balanced",
            'texture': "smooth"
        }

    # Анализ цветовой палитры с адаптивным количеством цветов
    color_thief = ColorThief(str(img_path))
    palette = color_thief.get_palette(color_count=6, quality=3)
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]

    # Анализ текстуры и стиля
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / (gray.size / 100)

    # Определение характеристик
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
        'symmetry': "balanced",  # Используем сбалансированную, а не зеркальную симметрию
        'texture': texture_type
    }


def generate_enhanced_pattern():
    # Инициализация модели с оптимизацией
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
            MODEL_ID,
            subfolder="scheduler"
        )
    ).to(device)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("Xformers не доступен, работаем без него")

    pipe.enable_attention_slicing()

    # Анализ исходного изображения
    design = analyze_design(INPUT_IMAGE)
    print(f"Анализ узора:")
    print(f"- Стиль: {design['style']}")
    print(f"- Сложность: {design['complexity']}")
    print(f"- Текстура: {design['texture']}")
    print(f"- Основные цвета: {', '.join(design['colors'][:4])}")

    # Генерация промпта с акцентом на естественные вариации
    prompt = f"""
Professional {design['style']} pattern design:
- Color palette: {", ".join(design['colors'])}
- {design['complexity']} complexity with natural variations
- Balanced composition without mirror symmetry
- Seamless tiling capability
- High-quality {design['texture']} texture
- Subtle organic imperfections
- 8K resolution details
- Harmonious color transitions
- Professional textile/wallpaper design
- Consistent but not repetitive elements
"""
    negative_prompt = """
mirrored, symmetric, duplicated, repetitive, 
blurry, low-res, artifacts, watermark, text, 
logo, cropped, deformed, disfigured, ugly, 
tiled, grid, chessboard, perfect symmetry
"""

    # Подготовка изображения
    init_image = Image.open(INPUT_IMAGE).convert("RGB")
    target_size = 768 if device == "cuda" else 512
    init_image = init_image.resize((target_size, target_size), Image.LANCZOS)
    # Создаем папку для результатов, если ее нет
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Генерация нескольких вариантов
    for i in range(1, NUM_GENERATIONS + 1):
        print(f"\nГенерация варианта {i} из {NUM_GENERATIONS}")

        # Параметры генерации для естественных вариаций
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=0.45,
            guidance_scale=9.0,
            num_inference_steps=60,
            generator=torch.Generator(device).manual_seed(42 + i),  # Разные сиды для вариаций
            eta=0.9
        ).images[0]

        # Постобработка в зависимости от типа узора
        if design['style'] == 'geometric':
            result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        else:
            result = result.filter(ImageFilter.SMOOTH_MORE)
            result = result.filter(ImageFilter.DETAIL)

        # Сохранение результата с номером генерации
        output_path = OUTPUT_DIR / f"enhanced_pattern_v{i}.png"
        result.save(output_path, quality=95, subsampling=0)
        print(f"Улучшенный узор сохранён: {output_path}")

if __name__ == "__main__":
    generate_enhanced_pattern()