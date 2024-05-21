import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

def find_and_crop_tray(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Параметры для фильтрации по цвету
    h_min = 0
    s_min = 20
    v_min = 59
    h_max = 24
    s_max = 66
    v_max = 186

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # Фильтрация по цвету
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Находим прямоугольный контур, описывающий лоток
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Обрезаем изображение по прямоугольнику, описывающему контур
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_img = image[y:y + h, x:x + w]

        # Изменяем размер обрезанного изображения на 1000x1000
        cropped_img = cv2.resize(cropped_img, (1000, 1000))
        return cropped_img
    else:
        return None

def filter_contours(contours):
    if not contours:
        return []

    # Находим контур с наибольшей площадью
    max_area = max([cv2.contourArea(cnt) for cnt in contours])

    # Определяем минимальную площадь для сохранения контуров
    min_area = max_area / 3

    # Отбираем только те контуры, площадь которых больше min_area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return filtered_contours

def find_circles(image, h_min, s_min, v_min, h_max, s_max, v_max, bloor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Параметры для фильтрации цвета для обрезанного изображения
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    hsv_blurred = cv2.GaussianBlur(hsv, (bloor * 2 + 1, bloor * 2 + 1), 0)
    mask = cv2.inRange(hsv_blurred, lower_bound, upper_bound)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем пустое изображение для отрисовки контуров
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, filter_contours(contours), -1, (0, 255, 0), 2)

    # Объединяем исходное изображение с изображением контуров
    result_image = cv2.addWeighted(image, 1, contour_image, 1, 0)

    # Отображаем обрезанное изображение с контурами
    cv2.imshow("Cropped Image", result_image)

    disk_areas_mm2 = calculate_disk_area(contours, image)
    return disk_areas_mm2

def calculate_disk_area(contours, cropped_image):
    # Получаем размеры обрезанного изображения
    img_height_px, img_width_px, _ = cropped_image.shape

    # Переводим размер изображения в миллиметры из пикселей
    img_height_px
    img_width_px

    # Размеры лотка в миллиметрах
    tray_width_mm, tray_height_mm = 300, 300

    # Площадь лотка в миллиметрах
    tray_area_mm2 = tray_width_mm * tray_height_mm

    # Площадь обрезанного изображения в миллиметрах
    img_area_px = img_height_px * img_width_px

    # Коэффициент различия между реальной площадью и фото (кв.мм в 1 px)
    k = tray_area_mm2 / img_area_px

    # Вычисление реальной площади ватных дисков в миллиметрах
    disk_areas_mm2 = sum([cv2.contourArea(cnt) * k for cnt in contours])

    return disk_areas_mm2


def choose_file():
    file_path = filedialog.askopenfilename()
    return file_path


def start_prog():
    def load_image():
        file_path = choose_file()
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение. Пожалуйста, выберите другой файл.")

            # Параметры фильтрации цвета по умолчанию
            h_min, s_min, v_min = 0, 0, 188
            h_max, s_max, v_max = 179, 255, 236
            bloor = 0

            # Обрезаем лоток изображения
            cropped_img = find_and_crop_tray(img)
            if cropped_img is not None:
                # Находим ватные диски на обрезанном изображении и вычисляем их площадь
                result = find_circles(cropped_img, h_min, s_min, v_min, h_max, s_max, v_max, bloor)
                if (result):
                    label.config(text=f"Площадь ватных дисков в кв. мм: {result}")
                else:
                    label.config(text=f"Не удалось вычислить площадь")
                root.focus_force()  # Активация главного окна и вывод на передний план

    # Создание главного окна
    root = tk.Tk()
    root.title("Подсчёт площади ватных дисков.")

    # Установка фиксированного размера окна
    root.geometry("450x80")

    # Установка невидоизменяемого размера окна
    root.resizable(False, False)

    # Получение размеров экрана
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Расчет положения окна по центру экрана
    x = (screen_width - root.winfo_reqwidth()) // 2
    y = (screen_height - root.winfo_reqheight()) // 2

    # Установка положения окна
    root.geometry("+{}+{}".format(x, y))

    # Создание метки для вывода результата
    label = tk.Label(root, text="")
    label.pack()

    # Создание кнопки для выбора файла
    button = tk.Button(root, text="Выбрать файл", command=load_image)
    button.pack()

    # Запуск главного цикла обработки событий
    root.mainloop()

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_prog()
