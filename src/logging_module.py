# -*- coding: utf-8 -*-
"""
Koib-V-4.2 — Модуль логирования запросов и ответов
=====================================================
Структурированное журналирование в JSONL формате:
  - timestamp, query_hash, model_type
  - вопрос, ответ, использованные чанки
  - результаты всех проверок валидации
  - итоговый статус (approved / review / rejected)
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("koib.query_logger")


class QueryLogger:
    """
    Логгер запросов и ответов в формате JSONL.
    
    Каждый запрос записывается как отдельная JSON-строка.
    Логирование работает даже при отказе отдельных модулей.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Args:
            log_dir: Директория для логов (по умолчанию output/logs)
        """
        self.log_dir = log_dir or Path(__file__).parent.parent / "output" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Формируем имя файла с датой
        self._current_date = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"queries_{self._current_date}.jsonl"
        
        # Проверяем, нужно ли создать новый файл (новая дата)
        self._check_date_rotation()
    
    def _check_date_rotation(self) -> None:
        """Проверить смену даты и создать новый файл при необходимости."""
        current_date = datetime.now().strftime("%Y%m%d")
        if current_date != self._current_date:
            self._current_date = current_date
            self.log_file = self.log_dir / f"queries_{self._current_date}.jsonl"
            logger.info(f"Создан новый файл лога: {self.log_file}")
    
    def _compute_query_hash(self, query: str) -> str:
        """Вычислить хэш запроса для идентификации дублей."""
        return hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
    
    def log(
        self,
        query: str,
        answer: str,
        model_type: str = "",
        sources: List[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        status: str = "approved",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Записать запрос и ответ в лог.
        
        Args:
            query: Текст вопроса пользователя
            answer: Текст ответа от системы
            model_type: Модель КОИБ (если указана)
            sources: Список источников (документы, страницы)
            validation_result: Результат валидации (из validation.py)
            status: Итоговый статус (approved | review | rejected)
            extra_metadata: Дополнительные метаданные
        
        Returns:
            True при успехе, False при ошибке
        """
        try:
            self._check_date_rotation()
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query_hash": self._compute_query_hash(query),
                "query": query,
                "model_type": model_type,
                "answer": answer,
                "sources": sources or [],
                "validation": validation_result or {},
                "status": status,
            }
            
            if extra_metadata:
                entry["metadata"] = extra_metadata
            
            # Записываем как JSONL (одна строка = один JSON-объект)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.debug(f"Запрос залогирован: {entry['query_hash']}")
            return True
            
        except Exception as exc:
            logger.error(f"Ошибка записи в лог: {exc}")
            # Не выбрасываем исключение — логирование не должно ломать основной пайплайн
            return False
    
    def get_log_path(self) -> Path:
        """Вернуть путь к текущему файлу лога."""
        return self.log_file
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Прочитать последние N записей из лога.
        
        Args:
            limit: Количество записей
        
        Returns:
            Список словарей с записями
        """
        entries = []
        try:
            if not self.log_file.exists():
                return entries
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
            
            # Возвращаем в обратном порядке (сначала новые)
            entries.reverse()
            
        except Exception as exc:
            logger.warning(f"Ошибка чтения лога: {exc}")
        
        return entries


# ═══════════════════════════════════════════════════════════════
# Глобальный экземпляр логгера
# ═══════════════════════════════════════════════════════════════

_global_logger: Optional[QueryLogger] = None


def get_query_logger() -> QueryLogger:
    """Получить глобальный экземпляр QueryLogger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = QueryLogger()
    return _global_logger


def log_query(
    query: str,
    answer: str,
    model_type: str = "",
    sources: List[Dict[str, Any]] = None,
    validation_result: Optional[Dict[str, Any]] = None,
    status: str = "approved",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Удобная функция для логирования запроса.
    
    Это обёртка над QueryLogger.log() для использования без импорта класса.
    """
    logger_instance = get_query_logger()
    return logger_instance.log(
        query=query,
        answer=answer,
        model_type=model_type,
        sources=sources,
        validation_result=validation_result,
        status=status,
        extra_metadata=extra_metadata,
    )
