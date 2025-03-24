"""Настройка логирования для проекта Tokenizator"""

import logging
import sys
import os


def setup_logger(log_file='tokenizator_model.log'):
    """Настройка логирования в файл и консоль"""
    # Создание директории для логов, если она не существует
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Настройка формата логирования
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Настройка корневого логгера
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Обработчик для вывода в файл
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    return logger


def get_logger(name):
    """Получение именованного логгера"""
    return logging.getLogger(name)