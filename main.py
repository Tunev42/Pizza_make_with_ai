# --- Импорты ---
import time
import math
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import re
import sys
from PIL import Image, ImageDraw, ImageFont
import io
import re
import asyncio
import random
from random import randint

try:
    from luckytools import LuckyTools
    tools = LuckyTools(prefix_short="⌊ Tatsumaki ⌉ » ", show_init=False)
except ImportError:
    class LuckyToolsStub:
        def fade_print(self, text, **kwargs):
            print(text)
    tools = LuckyToolsStub()

def get_val(item, k):
    for key, v in item.items():
        if key.strip() == k:
            return v
    return None

class PopularityNetwork:
    def __init__(self):
        self.lr_speed = 0.17
        random.seed(int(time.time()))
        # скрытый слой h1
        self.w11 = random.uniform(-1, 1)
        self.w12 = random.uniform(-1, 1)
        self.b1 = random.uniform(-1, 1)
        # скрытый слой h2
        self.w21 = random.uniform(-1, 1)
        self.w22 = random.uniform(-1, 1)
        self.b2 = random.uniform(-1, 1)

        self.w_out_1 = random.uniform(-1, 1)
        self.w_out_2 = random.uniform(-1, 1)
        self.b_out = random.uniform(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        # считаем скрытые слой
        self.h1_input = (x1 * self.w11) + (x2 * self.w12)
        self.h1_out = self.sigmoid(self.h1_input)

        self.h2_input = (x1 * self.w21) + (x2 * self.w22)
        self.h2_out = self.sigmoid(self.h2_input)

        # считаем выходные слои
        self.out_input = (self.h1_out * self.w_out_1) + (self.h2_out * self.w_out_2)
        self.final_out = self.sigmoid(self.out_input)

        return self.final_out

    def train(self, target):

        #ошибка выходного слоя
        out_error = target - self.final_out
        out_delta = out_error * self.sigmoid_derivative(self.final_out)

        # ошибка на скрытых слоях
        h1_error = out_delta * self.w_out_1
        h1_delta = h1_error * self.sigmoid_derivative(self.h1_out)

        h2_error = out_delta * self.w_out_2
        h2_delta = h2_error * self.sigmoid_derivative(self.h2_out)

        # корректировка весов выходного слоя
        self.w_out_1 += self.lr_speed * out_delta * self.h1_out
        self.w_out_2 += self.lr_speed * out_delta * self.h2_out
        self.b_out += self.lr_speed * out_delta

        # корректировка скрытых слоев
        self.w11 += self.lr_speed * h1_delta * self.x1
        self.w12 += self.lr_speed * h1_delta * self.x2
        self.b1 += self.lr_speed * h1_delta

        self.w21 += self.lr_speed * h2_delta * self.x1
        self.w22 += self.lr_speed * h2_delta * self.x2
        self.b2 += self.lr_speed * h2_delta


net = PopularityNetwork()

# кэш рейтингов, чтобы не парсить гугл каждый раз
ratings_cache = {}


def load_ratings_cache():
    global ratings_cache
    try:
        with open("ratings_cache.json", "r", encoding="utf-8") as file:
            ratings_cache = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        ratings_cache = {}


def save_ratings_cache():
    try:
        with open("ratings_cache.json", "w", encoding="utf-8") as file:
            json.dump(ratings_cache, file, indent=4, ensure_ascii=False)
    except:
        pass


def search_rating_online(dish_name):
    try:
        query = f"{dish_name} суши рейтинг отзывы"
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            matches = re.findall(r'(\d+\.\d+)\s*(?:из\s*5|звезд|рейтинг|оценка)', text, re.IGNORECASE)
            if matches:
                rating = float(matches[0])
                if 0 <= rating <= 10:
                    return rating
        return None
    except:
        return None


def get_rating(dish_name, price, menu_data):
    global ratings_cache
    if dish_name in ratings_cache:
        return ratings_cache[dish_name]
    # сначала смотрим в самом json
    for item in menu_data:
        name = get_val(item, "name")
        rating = get_val(item, "rating")
        if name and str(name).strip() == str(dish_name).strip():
            if rating is not None:
                ratings_cache[dish_name] = float(rating)
                save_ratings_cache()
                return float(rating)
    # если нет - идем в гугл
    rating = search_rating_online(dish_name)
    if rating is not None:
        ratings_cache[dish_name] = rating
        save_ratings_cache()
        return rating
    ratings_cache[dish_name] = None
    save_ratings_cache()
    return None


def preload_ratings(menu):
    tools.fade_print("парсим рейтинги, подожди", white_tag=True, time_show=0.01, color="b5b5b5")
    # TODO: потом сделать прогресс бар
    for item in menu:
        name = get_val(item, "name")
        price = get_val(item, "price_rub")
        if name:
            get_rating(str(name).strip(), float(price or 0), menu)
    save_ratings_cache()


def normalize_price(price, min_price=100, max_price=2000):
    return max(0, min(1, (price - min_price) / (max_price - min_price)))


def normalize_rating(rating, min_rating=0, max_rating=10):
    if rating is None:
        return 0.5
    return max(0, min(1, (rating - min_rating) / (max_rating - min_rating)))


def train_popularity_network():
    # данные для обучения: цена, рейтинг, популярность
    # логика такая: дешевое + высокий рейтинг = хит
    popularity_data = [
        (0.1, 0.9, 0.93), (0.2, 0.8, 0.87), (0.3, 0.7, 0.71),
        (0.5, 0.5, 0.5), (0.7, 0.3, 0.28), (0.85, 0.2, 0.13),
        (0.9, 0.1, 0.1), (0.4, 0.95, 0.8),
    ]
    tools.fade_print("Обучение нейросети для анализа популярности блюд...", white_tag=True, time_show=0.01,
                     color="b5b5b5")
    for epoc in range(10000):
        for x1, x2, target in popularity_data:
            net.forward(x1, x2)
            net.train(target)
    tools.fade_print("Нейросеть обучена", white_tag=True, time_show=0.5, color="00ff00")


def predict_popularity(price, rating):
    return net.forward(normalize_price(price), normalize_rating(rating))


def is_hit(rating, popularity_score):
    # без рейтинга хитом быть не может
    if rating is None:
        return False
    return popularity_score > 0.75


def get_popularity_label(rating, popularity_score):
    if is_hit(rating, popularity_score):
        return "ХИТ"
    elif popularity_score > 0.5:
        return "Популярное"
    elif popularity_score > 0.3:
        return "Среднее"
    else:
        return "Редко"


# --- Переменные ---
ingridients = {
    "wrapper": {
        "Нори": 5,
        "Лосось": 75,
        "Икра": 50,
        "Кунжут": 30,
        "Огурец": 20,
        "Пропустить": 0
    },
    "rice": {
        "Обычный": 20,
        "Солёный": 25
    },
    "filling": {
        "Творожный сыр": 30,
        "Огурецы": 15,
        "Лук": 15,
        "Лосось": 50,
        "Авокадо": 40,
        "Креветка": 80,
        "Угорь": 100,
        "Манго": 50
    }
}

menu_18 = [
    {"name": "Ролл 'Жаркий Чили'", "description": "С угрем, спайси-соусом и перцем халапеньо.", "price_rub": 450.0},
    {"name": "Сет 'Выходной'", "description": "Смелый выбор для ценителей. Много лосося, много огня.",
     "price_rub": 1800.0}
]

# что люди обычно докидывают в корзину (типа "другие посетители тоже купили")
# в "филадельфи" опечатка - не сработает на "филадельфия", но ладно
combo_rules = {
    "фри": ["Соус Сырный", "Кетчуп", "Соус Барбекю"],
    "наггетсы": ["Соус Сырный", "Кетчуп", "Соус Барбекю"],
    "палочки": ["Соус Сырный", "Кетчуп"],  # забыл барбекю добавить
    "шарики": ["Соус Сырный", "Кетчуп", "Соус Барбекю"],
    "филадельфи": ["Имбирь", "Васаби", "Соус Соевый"],  # опечатка, не сработает на "филадельфия"
    "том-ям": ["Добрый Кола 300мл", "Чай Rich зеленый"],
    "пицца": ["Добрый Лимон-лайм 300мл", "Добрый Кола 300мл", "Чизкейк Нью-Йорк"],
    "поке": ["Добрый Кола 300мл", "Чай Rich зеленый"]
}


def get_recommendations(cart, menu):
    recommendations = []
    cart_names_lower = [str(get_val(item, "name")).strip().lower() for item in cart]
    suggested_names = set()
    for cart_name in cart_names_lower:
        for keyword, suggestions in combo_rules.items():
            if keyword in cart_name:
                for sugg in suggestions:
                    suggested_names.add(sugg)
    # если закуски есть, а пить нечего - предложим колу
    has_snack = any("закуск" in str(get_val(item, "category")).lower() for item in cart)
    has_drink = any("напитк" in str(get_val(item, "category")).lower() for item in cart)
    if has_snack and not has_drink:
        suggested_names.add("Добрый Кола 300мл")
    for item in menu:
        item_name = str(get_val(item, "name")).strip()
        if item_name in suggested_names:
            already_in_cart = any(str(get_val(c, "name")).strip() == item_name for c in cart)
            if not already_in_cart:
                recommendations.append(item)
    return recommendations[0:3]


# --- Декораторы ---
def require_working_hours(func):
    def wrapper(*args, **kwargs):
        current_hour = int(str(datetime.now()).split(" ")[1].split(":")[0])
        if 7 <= current_hour < 23:
            return func(*args, **kwargs)
        tools.fade_print(f"✕ Мы работаем с 7:00 до 23:00", white_tag=True, time_show=3,
                         color="ff0000")
        return None

    return wrapper


# --- Утилиты ---
def shitaem_total_recursive(cart, index=0):
    if index == len(cart):
        return 0
    return float(get_val(cart[index], "price_rub") or 0) + shitaem_total_recursive(cart, index + 1)


async def process_order():
    tools.fade_print(f"⟳ Установка защищенного соединения...", white_tag=True, time_show=0.01, color="b5b5b5")
    await asyncio.sleep(1)
    tools.fade_print(f"⟳ Обработка...", white_tag=True, time_show=0.01, color="b5b5b5")
    await asyncio.sleep(1.5)
    tools.fade_print("✓ Транзакция подтверждена сервером", white_tag=True, time_show=0.5, color="00ff00")


def get_captcha(width=200,
                height=100):  # https://docs-python.ru/packages/veb-frejmvork-flask-python/generatsiia-svoei-kapchi/
    code = ''.join([random.choice('АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ') for i in range(5)])

    img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('freesans.ttf', size=50)
    except IOError:
        font = ImageFont.load_default()
    x = 0;
    y = 12;
    for let in code:
        if x == 0:
            x = 5
        else:
            x = x + width / len(code)
        y = randint(3, 55)
        draw.text((x, y), let, font=font, fill=(randint(0, 200), randint(0, 200), randint(0, 200), 128))

    for i in range(40):
        draw.line([(randint(0, width), randint(0, height)),
                   (randint(0, width), randint(0, height))],
                  randint(0, 200), 2, 128)

    img.save("captha.png", "PNG")
    return code


def new_input(prompt_text, max_val=sys.maxsize, min_val=1):
    try:
        answer = input("› ")
        int_answer = int(answer)

        if min_val <= int_answer <= max_val:
            return int_answer

        tools.fade_print(f"✕ Ошибка ввода! Введите число от {min_val} до {max_val}.", white_tag=True, time_show=1.5,
                         color="ff0000")
    except ValueError:
        tools.fade_print("✕ Ошибка ввода! Попробуйте ещё раз.", white_tag=True, time_show=1.5, color="ff0000")
    return new_input(prompt_text, max_val, min_val)


def get_age_and_name():
    try:
        tools.fade_print("Импорт данных меню из JSON...", white_tag=True, time_show=0.01)
        with open("info.json", "r", encoding="utf-8") as file:
            information = json.load(file)
        tools.fade_print("✓ Импорт успешно завершён", white_tag=True, time_show=0.01)
    except (FileNotFoundError, json.JSONDecodeError):
        tools.fade_print("✕ JSON не найден или повреждён.",
                         white_tag=True, time_show=1.5, color="ffb300")
        with open("info.json", "w", encoding="utf-8") as file:
            info_def = {
                "admin@tatsumaki.xyz": {
                    "name": "Администратор",
                    "age": 21,
                    "phone": "79999999999",
                    "password": "admin"
                }
            }
            json.dump(info_def, file, indent=4, ensure_ascii=False)
            information = info_def

    while True:
        tools.fade_print("Пожалуйста введите своё ФИО или 1, для входа:", white_tag=True, time_show=0.01)
        name = input("› ")
        if name == "1":
            tools.fade_print("Пожалуйста введите свой email:", white_tag=True,
                             time_show=0.01)
            email = input("› ").lower()
            if email in information:
                user_data = information[email]

                tools.fade_print("Пожалуйста введите пароль:", white_tag=True, time_show=0.01)
                password_input = input("› ")

                if user_data.get("password") == password_input:
                    return user_data["age"], user_data["name"], user_data["phone"], email
                else:
                    tools.fade_print("✕ Неверный пароль.", white_tag=True, time_show=3, color="ff0000")
                    continue
            else:
                tools.fade_print("✕ Почта не найдена.",
                                 white_tag=True,
                                 time_show=3, color="ff0000")
                continue

        age = None
        while age is None:
            try:
                tools.fade_print("Пожалуйста введите свой возраст (только число):", white_tag=True, time_show=0.01)
                age_input = int(input("› "))
                if 3 < age_input < 123:
                    age = age_input
                else:
                    tools.fade_print("✕ Возраст должен быть в разумных пределах (4-122). Попробуйте ещё раз.",
                                     white_tag=True,
                                     time_show=3, color="ff0000")
            except ValueError:
                tools.fade_print("✕ Ошибка ввода! Попробуйте ещё раз.", white_tag=True, time_show=3, color="ff0000")

        number = None
        while number is None:
            tools.fade_print("Пожалуйста введите свой номер телефона:", white_tag=True,
                             time_show=0.01)
            number_raw = input("› ").strip()

            if bool(re.fullmatch(r"^((8|\+7)[\- ]?)?(\(?\d{3}\)?[\- ]?)?[\d\- ]{7,10}$", number_raw)):
                normalized_number = re.sub(r"[^\d]", "", number_raw)
                if normalized_number.startswith("8"):
                    normalized_number = "7" + normalized_number[1:]
                elif normalized_number.startswith("+7"):
                    normalized_number = normalized_number[1:]

                if len(normalized_number) == 11 and normalized_number.startswith("7"):
                    number = normalized_number
                else:
                    tools.fade_print("✕ Номер телефона введен неверно. Поддерживается только +7/8.",
                                     white_tag=True,
                                     time_show=3, color="ff0000")
            else:
                tools.fade_print("✕ Номер телефона введен неверно. Поддерживается только +7/8.",
                                 white_tag=True,
                                 time_show=3, color="ff0000")

        email = None
        while email is None:
            tools.fade_print("Пожалуйста введите свой email:", white_tag=True,
                             time_show=0.01)
            email_input = input("› ").lower()
            if bool(re.fullmatch(
                    r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""
                    , email_input)):
                if email_input in information:
                    tools.fade_print("✕ Эта почта уже зарегистрирована. Попробуйте войти.",
                                     white_tag=True, time_show=3, color="ffb300")
                    break
                tools.fade_print("Придумайте пароль для входа:", white_tag=True, time_show=0.01)
                password_input = input("› ")
                email = email_input
                information[email] = {
                    "name": name.title(),
                    "age": age,
                    "phone": number,
                    "password": password_input
                }
                try:
                    with open("info.json", "w", encoding="utf-8") as file:
                        json.dump(information, file, indent=4, ensure_ascii=False)
                except IOError:
                    tools.fade_print("✕ Ошибка сохранения данных пользователя.",
                                     white_tag=True, time_show=3, color="ff0000")

                return age, name, number, email
            else:
                tools.fade_print(f"✕ Почта введена неверно.",
                                 white_tag=True,
                                 time_show=3, color="ff0000")


def display_image_from_url(url, w=180):
    def pil_to_ascii_color(image):  # Нашел эту штуку в интернете
        char = "█"  # https://ru.piliapp.com/symbol/square/
        W = w

        width, height = image.size
        aspect_ratio = height / width
        new_height = int(W * aspect_ratio * 0.55)
        resized_image = image.resize((W, new_height))

        pixels = resized_image.getdata()
        output_string = []

        def get_color_code(r, g, b):
            r_i = int(r / 255 * 5)
            g_i = int(g / 255 * 5)
            b_i = int(b / 255 * 5)
            color_index = 16 + (r_i * 36) + (g_i * 6) + b_i
            return f"\033[38;5;{color_index}m"

        for i, pixel in enumerate(pixels):
            r, g, b = pixel[:3]
            color_code = get_color_code(r, g, b)
            output_string.append(f"{color_code}{char}")
            if (i + 1) % W == 0:
                output_string.append("\033[0m\n")

        return "".join(output_string) + "\033[0m"

    try:
        if not url:
            tools.fade_print("| (Нет ссылки на изображение)", white_tag=True, time_show=0.01, color="b5b5b5")
            return

        if str(url).startswith("http"):
            response = requests.get(url, stream=True, timeout=5)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(url).convert("RGB")
        ascii_art = pil_to_ascii_color(image)
        print(ascii_art)

    except requests.exceptions.RequestException:
        tools.fade_print("| Загрузка... ---", white_tag=True, time_show=0.01, color="ffb300")
    except Exception:
        tools.fade_print("| Ошибка обработки изображения с помощью PIL.", white_tag=True, time_show=0.01,
                         color="ffb300")


def open_constructor(menu, cart, age, balance, bonus):
    tools.fade_print("⇌ Конструктор роллов", white_tag=True, time_show=0.01, color="ffffff")
    price = 0
    new_roll_details = {}

    for section_name, ingredients in ingridients.items():
        list_ingredients = list(ingredients.items())
        tools.fade_print(f"⇌ Выберите {section_name} (Текущая цена: {price} руб):", white_tag=True, time_show=0.01,
                         color="ffffff")

        for i, (name, cost) in enumerate(list_ingredients):
            tools.fade_print(f"| {i + 1}. {name} - {cost} руб.", white_tag=True, time_show=0.01, color="b5b5b5")

        tools.fade_print(f"| {len(list_ingredients) + 1}. Назад в меню", white_tag=True, time_show=0.01, color="b5b5b5")

        selection_index = new_input("", max_val=len(list_ingredients) + 1)

        if selection_index == len(list_ingredients) + 1:
            return main_menu(menu, cart, age, balance, bonus)

        selected_item = list_ingredients[selection_index - 1]
        new_roll_details[section_name] = selected_item
        price += selected_item[1]

    description_text = (
        f"Обертка: {new_roll_details['wrapper'][0]} ({new_roll_details['wrapper'][1]} руб), "
        f"Рис: {new_roll_details['rice'][0]} ({new_roll_details['rice'][1]} руб), "
        f"Начинка: {new_roll_details['filling'][0]} ({new_roll_details['filling'][1]} руб)."
    )

    cart.append({
        "product_id": "0000",
        "name": f"Кастомный ролл ({new_roll_details['wrapper'][0]})",
        "category": "Конструктор роллов",
        "description": description_text,
        "image_url": None,
        "weight": "250 гр",
        "price_rub": float(price),
        "rating": None
    })

    tools.fade_print(f"Итоговая цена: {price} рублей. ✓ Успешно добавлено в корзину.", white_tag=True, time_show=2,
                     color="ffffff")
    return main_menu(menu, cart, age, balance, bonus)


def open_hits_menu(menu, cart, age, balance, bonus):
    print("\n" * 2)
    tools.fade_print("!!! ХИТЫ ПРОДАЖ !!!", white_tag=True, time_show=0.01, color="ff0000")

    hits = []
    for item in menu:
        roll_name = str(get_val(item, "name")).strip()
        price = float(get_val(item, "price_rub") or 0)
        rating = get_rating(roll_name, price, menu)
        popularity_score = predict_popularity(price, rating)
        if is_hit(rating, popularity_score):
            hits.append({"item": item, "name": roll_name, "price": price, "rating": rating,
                         "popularity_score": popularity_score})

    if not hits:
        tools.fade_print("Хитов пока нет", white_tag=True, time_show=2, color="b5b5b5")
        return main_menu(menu, cart, age, balance, bonus)

    tools.fade_print("Выберите хит:", white_tag=True, time_show=0.01, color="ffffff")
    for i, roll in enumerate(hits):
        rating_str = f"{roll['rating']:.1f}" if roll["rating"] is not None else "Нет"
        tools.fade_print(f"| {i + 1}. {roll['name']} | {roll['price']} руб. | {rating_str}", white_tag=True,
                         time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| {len(hits) + 1}. Назад в меню", white_tag=True, time_show=0.01, color="b5b5b5")

    roll_index = new_input("", max_val=len(hits) + 1)
    if roll_index == len(hits) + 1:
        return main_menu(menu, cart, age, balance, bonus)

    selected = hits[roll_index - 1]
    print("\n")
    display_image_from_url(get_val(selected["item"], "image_url"))

    tools.fade_print(f"{selected['name']}", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"| Категория: {get_val(selected['item'], 'category')}", white_tag=True, time_show=0.01,
                     color="b5b5b5")
    tools.fade_print(f"| Описание: {get_val(selected['item'], 'description')}", white_tag=True, time_show=0.01,
                     color="b5b5b5")
    tools.fade_print(f"| Вес: {get_val(selected['item'], 'weight')}", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| Рейтинг: {selected['rating'] if selected['rating'] is not None else 'Нет данных'}",
                     white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(
        f"| AI-анализ: {get_popularity_label(selected['rating'], selected['popularity_score'])} (уверенность: {selected['popularity_score']:.0%})",
        white_tag=True, time_show=0.01, color="00ff00")
    tools.fade_print(f"|-- Цена: {selected['price']} руб. ------", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 1. Добавить в корзину", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 2. Назад в главное меню", white_tag=True, time_show=0.01, color="b5b5b5")

    action_choice = new_input("", max_val=2)

    if action_choice == 1:
        cart.append(selected["item"])
        tools.fade_print(f"✓ Успешно добавлено в корзину.", white_tag=True, time_show=1.5)
    return main_menu(menu, cart, age, balance, bonus)


def open_menu_standart(menu, cart, age, balance, bonus):
    print("\n" * 2)

    categories = []
    for item in menu:
        cat = get_val(item, "category")
        if cat:
            cat_clean = str(cat).strip()
            if not cat_clean in categories:
                categories.append(cat_clean)

    tools.fade_print("⇌ Выберите категорию:", white_tag=True, time_show=0.01, color="ffffff")
    for i, cat in enumerate(categories):
        tools.fade_print(f"| {i + 1}. {cat}", white_tag=True, time_show=0.01, color="b5b5b5")

    tools.fade_print(f"| {len(categories) + 1}. Назад в меню", white_tag=True, time_show=0.01, color="b5b5b5")

    selection_index = new_input("", max_val=len(categories) + 1)
    if selection_index == len(categories) + 1:
        return main_menu(menu, cart, age, balance, bonus)

    selected_category = categories[selection_index - 1]

    # хиты отсюда фильтруем, они только в разделе ХИТЫ
    filtered_rolls = []
    for item in menu:
        cat = get_val(item, "category")
        if cat and str(cat).strip() == selected_category:
            roll_name = str(get_val(item, "name")).strip()
            price = float(get_val(item, "price_rub") or 0)
            rating = get_rating(roll_name, price, menu)
            popularity_score = predict_popularity(price, rating)
            if not is_hit(rating, popularity_score):
                filtered_rolls.append({"item": item, "name": roll_name, "price": price, "rating": rating,
                                       "popularity_score": popularity_score})

    print("\n" * 1)
    tools.fade_print(f"⇌ Выберите блюдо из категории '{selected_category}':", white_tag=True, time_show=0.01,
                     color="ffffff")
    for i, roll in enumerate(filtered_rolls):
        rating_str = f"{roll['rating']:.1f}" if roll["rating"] is not None else "Нет"
        tools.fade_print(
            f"| {i + 1}. {roll['name']} | {roll['price']} руб. | {rating_str} | {get_popularity_label(roll['rating'], roll['popularity_score'])}",
            white_tag=True, time_show=0.01, color="b5b5b5")

    tools.fade_print(f"| {len(filtered_rolls) + 1}. Назад к категориям", white_tag=True, time_show=0.01,
                     color="b5b5b5")

    roll_index = new_input("", max_val=len(filtered_rolls) + 1)
    if roll_index == len(filtered_rolls) + 1:
        return open_menu_standart(menu, cart, age, balance, bonus)

    selected = filtered_rolls[roll_index - 1]

    print("\n")

    display_image_from_url(get_val(selected["item"], "image_url"))

    tools.fade_print(f"⇌ {selected['name']}", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"| Категория: {selected_category}", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| Описание: {get_val(selected['item'], 'description')}", white_tag=True, time_show=0.01,
                     color="b5b5b5")
    tools.fade_print(f"| Вес: {get_val(selected['item'], 'weight')}", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| Рейтинг: {selected['rating'] if selected['rating'] is not None else 'Нет данных'}",
                     white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(
        f"| AI-анализ: {get_popularity_label(selected['rating'], selected['popularity_score'])} (уверенность: {selected['popularity_score']:.0%})",
        white_tag=True, time_show=0.01, color="00ff00")
    tools.fade_print(f"|-- Цена: {selected['price']} руб. ------", white_tag=True, time_show=0.01, color="b5b5b5")

    tools.fade_print(f"| 1. Добавить в корзину", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 2. Назад в главное меню", white_tag=True, time_show=0.01, color="b5b5b5")

    action_choice = new_input("", max_val=2)

    if action_choice == 1:
        cart.append(selected["item"])
        tools.fade_print(f"✓ Успешно добавлено в корзину.", white_tag=True, time_show=1.5)
        return main_menu(menu, cart, age, balance, bonus)
    else:
        return main_menu(menu, cart, age, balance, bonus)


def open_adult_menu(menu, cart, age, balance, bonus):
    print("\n" * 2)
    tools.fade_print("!!! ВЗРОСЛОЕ МЕНЮ (18+) !!!", white_tag=True, time_show=0.01, color="ff0000")
    tools.fade_print("⇌ Выберите блюдо:", white_tag=True, time_show=0.01, color="ffffff")

    for i, item in enumerate(menu_18):
        tools.fade_print(f"| {i + 1}. {item['name']} ({item['description']}) - {item['price_rub']} руб. | 5.0",
                         white_tag=True, time_show=0.01, color="b5b5b5")

    tools.fade_print(f"| {len(menu_18) + 1}. Назад в меню", white_tag=True, time_show=0.01, color="b5b5b5")

    roll_index = new_input("", max_val=len(menu_18) + 1)
    if roll_index == len(menu_18) + 1:
        return main_menu(menu, cart, age, balance, bonus)

    selected_item = menu_18[roll_index - 1].copy()

    selected_item["product_id"] = f"A{roll_index:03d}"
    selected_item["category"] = "Взрослое меню (18+)"
    selected_item["weight"] = "Различный"
    selected_item["image_url"] = None
    selected_item["rating"] = 5.0

    cart.append(selected_item)
    tools.fade_print(f"✓ '{selected_item['name']}' успешно добавлено в корзину.", white_tag=True, time_show=2)
    return main_menu(menu, cart, age, balance, bonus)


def open_menu(menu, cart, age, balance, bonus):
    print("\n" * 2)
    tools.fade_print("⇌ Выберите раздел меню:", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print("| 1. ХИТЫ", white_tag=True, time_show=0.01, color="ff0000")
    tools.fade_print("| 2. Основное меню", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print("| 3. Конструктор роллов", white_tag=True, time_show=0.01, color="b5b5b5")

    option_count = 3
    if age >= 18:
        option_count += 1
        tools.fade_print("| 4. Взрослое меню (18+)", white_tag=True, time_show=0.01, color="ff0000")

    tools.fade_print(f"| {option_count + 1}. Назад в главное меню", white_tag=True, time_show=0.01, color="b5b5b5")

    while True:
        answer = new_input("", max_val=option_count + 1)

        if answer == 1:
            return open_hits_menu(menu, cart, age, balance, bonus)
        elif answer == 2:
            return open_menu_standart(menu, cart, age, balance, bonus)
        elif answer == 3:
            return open_constructor(menu, cart, age, balance, bonus)
        elif answer == 4 and age >= 18:
            return open_adult_menu(menu, cart, age, balance, bonus)
        elif answer == option_count + 1:
            return main_menu(menu, cart, age, balance, bonus)
        else:
            tools.fade_print("✕ Некорректный выбор. Попробуйте ещё раз.", white_tag=True, time_show=1.5, color="ff0000")


# --- Корзина и оплата ---
def pay_card(cart, final_price, balance):
    asyncio.run(process_order())
    print("\n" * 1)
    tools.fade_print("- Оформление заказа (Карта) -", white_tag=True, time_show=0.01, color="ffffff")

    for item in cart:
        tools.fade_print(f"| {get_val(item, 'name')} - {get_val(item, 'price_rub')} руб", white_tag=True,
                         time_show=0.01, color="b5b5b5")

    tools.fade_print(f"- Итого: {final_price} руб.", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"- Статус: Ожидание оплаты.", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"Для оплаты, переведите {final_price} рублей на номер:", white_tag=True, time_show=0.01,
                     color="b5b5b5")
    tools.fade_print(f"--- +7 923 504 68-72 ---", white_tag=True, time_show=0.01)
    tools.fade_print(f"После перевода, заказ будет принят. Мы свяжемся с вами для уточнения деталей.", white_tag=True,
                     time_show=0.01, color="b5b5b5")


def pay_cash(cart, final_price, balance):
    asyncio.run(process_order())
    print("\n" * 1)
    tools.fade_print("- Оформление заказа (Наличными) -", white_tag=True, time_show=0.01, color="ffffff")

    for item in cart:
        tools.fade_print(f"| {get_val(item, 'name')} - {get_val(item, 'price_rub')} руб", white_tag=True,
                         time_show=0.01, color="b5b5b5")

    tools.fade_print(f"- Итого: {final_price} руб.", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"- Статус: Оплата при получении.", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print("Ваш заказ принят. Ожидайте звонка оператора.", white_tag=True, time_show=0.01, color="b5b5b5")


def pay_wallet(cart, final_price, balance):
    if balance < final_price:
        tools.fade_print(f"✕ Недостаточно средств на кошельке! Баланс: {balance} руб., нужно: {final_price} руб.",
                         white_tag=True, time_show=3, color="ff0000")
        return balance
    asyncio.run(process_order())
    new_balance = balance - final_price
    print("\n" * 1)
    tools.fade_print("- Оформление заказа (Кошелек) -", white_tag=True, time_show=0.01, color="ffffff")

    for item in cart:
        tools.fade_print(f"| {get_val(item, 'name')} - {get_val(item, 'price_rub')} руб", white_tag=True,
                         time_show=0.01, color="b5b5b5")

    tools.fade_print(f"- Итого: {final_price} руб.", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"✓ Оплата прошла успешно!", white_tag=True, time_show=0.01, color="00ff00")
    tools.fade_print(f"- Остаток на кошельке: {new_balance} руб.", white_tag=True, time_show=0.01, color="ffffff")
    return new_balance


def open_cart(menu, cart, age, balance, bonus):
    print("\n" * 2)
    if not cart:
        tools.fade_print("⇌ Корзина пуста.", white_tag=True, time_show=0.01, color="ffffff")
        tools.fade_print("| 1. Назад", white_tag=True, time_show=0.01, color="b5b5b5")
        if new_input("", max_val=1) == 1:
            return main_menu(menu, cart, age, balance, bonus)
        return main_menu(menu, cart, age, balance, bonus)

    final_price = shitaem_total_recursive(cart)

    tools.fade_print("⇌ Ваша Корзина", white_tag=True, time_show=0.01, color="ffffff")
    for i, item in enumerate(cart):
        tools.fade_print(
            f"| {i + 1}. {get_val(item, 'name')} ({get_val(item, 'category')}) - {get_val(item, 'price_rub')} руб",
            white_tag=True, time_show=0.01, color="b5b5b5")

    tools.fade_print(f"| --- Итого: {final_price} руб ---------", white_tag=True, time_show=0.01, color="ffffff")

    recommendations = get_recommendations(cart, menu)
    if recommendations:
        print("\n")
        tools.fade_print("[!] Другие посетители часто добавляют к вашему заказу:", white_tag=True, time_show=0.01,
                         color="ffb300")
        for i, rec in enumerate(recommendations):
            rec_name = get_val(rec, "name")
            rec_price = get_val(rec, "price_rub")
            tools.fade_print(f"  {i + 1}. {rec_name} ({rec_price} руб.)", white_tag=True, time_show=0.01,
                             color="ffffff")

        tools.fade_print("Хотите добавить что-то из этого? (0 - нет, или введите номер)", white_tag=True,
                         time_show=0.01, color="b5b5b5")
        rec_choice = new_input("", max_val=len(recommendations), min_val=0)

        if rec_choice > 0:
            selected_rec = recommendations[rec_choice - 1]
            cart.append(selected_rec)
            tools.fade_print(f"✓ Успешно добавлено: '{get_val(selected_rec, 'name')}'", white_tag=True,
                             time_show=1.5, color="00ff00")
            return open_cart(menu, cart, age, balance, bonus)

    tools.fade_print(f"| 1. Оформить заказ", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 2. Назад", white_tag=True, time_show=0.01, color="b5b5b5")

    action_choice = new_input("", max_val=2)

    if action_choice == 2:
        return main_menu(menu, cart, age, balance, bonus)

    tools.fade_print("\n⇌ Выберите способ оплаты", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"| 1. Оплата картой", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 2. Наличными при получении", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 3. С кошелька (Баланс: {balance} руб.)", white_tag=True, time_show=0.01, color="b5b5b5")

    payment_choice = new_input("", max_val=3)

    if payment_choice == 1:
        pay_card(cart, final_price, balance)
    elif payment_choice == 2:
        pay_cash(cart, final_price, balance)
    elif payment_choice == 3:
        new_balance = pay_wallet(cart, final_price, balance)
        if new_balance != balance:
            i = []
            cart = i
            return main_menu(menu, cart, age, new_balance, bonus)

    return main_menu(menu, cart, age, balance, bonus)


# --- Основное меню ---
def main_menu(menu, cart, age, balance, bonus):
    print("\n" * 5)
    tools.fade_print("⇌ Главное меню", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print("| 1. Меню", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 2. Корзина ({len(cart)})", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 3. Моя KASHOLKA)", white_tag=True, time_show=0.01,
                     color="b5b5b5")

    max_option = 3

    answer = new_input("", max_val=max_option)

    if answer == 1:
        return open_menu(menu, cart, age, balance, bonus)
    elif answer == 2:
        return open_cart(menu, cart, age, balance, bonus)
    elif answer == 3:
        return open_cash(menu, cart, age, balance, bonus)

def open_cash(menu, cart, age, balance, bonus):
    tools.fade_print("⇌ Кошелек", white_tag=True, time_show=0.01, color="ffffff")
    tools.fade_print(f"| Баланс: {balance}", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| Бонусов: {bonus}", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 1. Назад", white_tag=True, time_show=0.01, color="b5b5b5")
    tools.fade_print(f"| 2. Пополнить счёт", white_tag=True, time_show=0.01, color='b5b5b5')
    x = input()

    while True:
        if x == '1':
            return main_menu(menu, cart, age, balance, bonus)
        elif x == '2':
            return add_value(menu, cart, age, balance, bonus)
        else:
            print("Введите '1' или '2'")


def add_value(menu, cart, age, balance, bonus):
    try:
        while True:
            num = int(input("Введите сколько вы хотите внести денег: "))
            if num > 0 and num < 999999999999999:
                balance += num
                break
            elif num < 0:
                print("Введите целое число, которое больше 0 и меньше 99999999")
                continue
            else:
                continue
    except ValueError as e:
        print(f"Ошбика кода {e}")

    return main_menu(menu, cart, age, balance, bonus)


@require_working_hours
def run_app():
    cart = []
    balance = 0
    bonus = 50

    user_data = get_age_and_name()
    if user_data is None:
        return

    age, name, phone, email = user_data

    menu = []
    try:
        tools.fade_print("Импорт данных меню из JSON...", white_tag=True, time_show=0.01)
        with open("menu.json", "r", encoding="utf-8") as file:
            menu = json.load(file)
        tools.fade_print("✓ Импорт успешно завершён", white_tag=True, time_show=0.01)
    except (FileNotFoundError, json.JSONDecodeError):
        tools.fade_print("✕ JSON не найден или повреждён.",
                         white_tag=True, time_show=1.5, color="ffb300")
        return

    preload_ratings(menu)
    main_menu(menu, cart, age, balance, bonus)


if __name__ == "__main__":
    load_ratings_cache()
    train_popularity_network()

    tools = LuckyTools(prefix_short="⌊ УниКапча ⌉ » ", show_init=False)
    while True:
        code = get_captcha(600, 100)
        display_image_from_url("captha.png")
        tools.fade_print("Введите код  или debug", white_tag=True, time_show=1)
        enter = input("> ")
        if enter.lower() == code.lower() or enter == "debug":
            tools.fade_print("Верный код, спасибо за прохождение проверки!", white_tag=True, time_show=1)
            tools = LuckyTools(prefix_short="⌊ Tatsumaki ⌉ » ", show_init=False)
            run_app()
        else:
            tools.fade_print("✕ Неверно введен код, попробуйте ещё раз!",
                             white_tag=True, time_show=1.5, color="ffb300")
