import os

# Пути к директориям
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'feature_cache')
IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, 'image_cache')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Настройки по умолчанию
DEFAULT_RANDOM_STATE = 42
DEFAULT_THRESHOLD = 50  # Порог для бинарной классификации
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'tokenizator_model.txt')
DEFAULT_DATA_FILE = 'tweet_dataset_attributes.json'

# Настройки моделей
BERTWEET_MODEL_NAME = 'vinai/bertweet-base'
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
EMOTION_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
EMOTION_LABELS = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
    'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
]

# Параметры обработки
MAX_TEXT_LENGTH = 128
BATCH_SIZE = 32
TEXT_PCA_COMPONENTS = 45
IMAGE_PCA_COMPONENTS = 60
CORRELATION_THRESHOLD = 0.8

# Создание необходимых директорий
for dir_path in [CACHE_DIR, IMAGE_CACHE_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
