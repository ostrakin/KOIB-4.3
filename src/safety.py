# -*- coding: utf-8 -*-
"""
Koib-V-4.2 — Модуль безопасности и детекции чувствительных тем
=================================================================
Определение запросов, требующих ручной проверки:
  - Жалобы на оборудование
  - Технические сбои
  - Недействительные бюллетени
  - Маркировка бюллетеней
  - Другие критические темы
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("koib.safety")


# ═══════════════════════════════════════════════════════════════
# Категории чувствительных тем
# ═══════════════════════════════════════════════════════════════

@dataclass
class SensitiveTopic:
    """Описание чувствительной темы."""
    category: str
    patterns: List[str]
    description: str
    requires_review: bool = True


# Список чувствительных тем с регулярными выражениями
SENSITIVE_TOPICS = [
    SensitiveTopic(
        category="invalid_ballot",
        patterns=[
            r"\bнедействительн(?:ый|ые|ых)\s+бюллетен(?:ь|и|ей)\b",
            r"\bнедействительност(?:ь|и)\s+бюллетен(?:я|ей)\b",
            r"\bпризна(?:ть|вать)\s+(?:бюллетень|голосование)\s+недействительным\b",
            r"\bаннулировани(?:е|я)\s+бюллетен(?:я|ей)\b",
        ],
        description="Вопросы о недействительных бюллетенях",
    ),
    SensitiveTopic(
        category="complaint",
        patterns=[
            r"\bжалоб(?:а|ы|у)\b",
            r"\bпожаловаться\b",
            r"\bпротест\b",
            r"\bнарушени(?:е|я|й)\b",
            r"\bнесогласи(?:е|я)\b",
            r"\bапелляци(?:я|и|ю)\b",
        ],
        description="Жалобы и нарушения",
    ),
    SensitiveTopic(
        category="technical_failure",
        patterns=[
            r"\bтехнический\s+сбой\b",
            r"\bнеисправност(?:ь|и)\b",
            r"\bполомка\b",
            r"\bотказ\s+(?:оборудования|терминала|КОИБ)\b",
            r"\bошибка\s+(?:сканирования|обработки|считывания)\b",
            r"\bзавис(?:ание|ания|ает)\b",
            r"\bперезагрузк(?:а|и|у)\b",
            r"\bне\s+работает\b",
            r"\bне\s+включается\b",
        ],
        description="Технические сбои и неисправности",
    ),
    SensitiveTopic(
        category="ballot_marking",
        patterns=[
            r"\bмаркировк(?:а|и|у)\s+бюллетен(?:я|ей)\b",
            r"\bпометк(?:а|и|и)\s+в\s+бюллетене\b",
            r"\bотметк(?:а|и|у)\s+избирател(?:я|ей)\b",
            r"\bзаполнени(?:е|я)\s+бюллетен(?:я|ей)\b",
        ],
        description="Маркировка и заполнение бюллетеней",
    ),
    SensitiveTopic(
        category="security",
        patterns=[
            r"\bбезопасност(?:ь|и)\b",
            r"\bвзлом\b",
            r"\bнесанкционированный\s+доступ\b",
            r"\bутечка\s+данных\b",
            r"\bфальсификаци(?:я|и|ю)\b",
        ],
        description="Вопросы безопасности",
    ),
    SensitiveTopic(
        category="manual_processing",
        patterns=[
            r"\brучной\s+подсчет\b",
            r"\brучная\s+обработка\b",
            r"\bручной\s+режим\b",
            r"\bрезервный\s+режим\b",
        ],
        description="Ручная обработка данных",
    ),
]


# ═══════════════════════════════════════════════════════════════
# Результат детекции
# ═══════════════════════════════════════════════════════════════

@dataclass
class SafetyCheckResult:
    """Результат проверки на чувствительные темы."""
    is_sensitive: bool
    detected_categories: List[str]
    matched_patterns: List[str]
    requires_human_review: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализовать в словарь."""
        return {
            "is_sensitive": self.is_sensitive,
            "detected_categories": self.detected_categories,
            "matched_patterns": self.matched_patterns,
            "requires_human_review": self.requires_human_review,
        }


# ═══════════════════════════════════════════════════════════════
# Детектор чувствительных тем
# ═══════════════════════════════════════════════════════════════

class SafetyDetector:
    """
    Детектор чувствительных тем в запросах.
    
    Автоматически помечает запросы, требующие ручной проверки.
    """
    
    def __init__(self, topics: Optional[List[SensitiveTopic]] = None):
        """
        Args:
            topics: Список тем для детекции (по умолчанию — SENSITIVE_TOPICS)
        """
        self.topics = topics or SENSITIVE_TOPICS
        # Компилируем паттерны для эффективности
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, SensitiveTopic]]] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Скомпилировать все регулярные выражения."""
        for topic in self.topics:
            compiled_for_topic = []
            for pattern_str in topic.patterns:
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                    compiled_for_topic.append((compiled, topic))
                except re.error as exc:
                    logger.warning(f"Ошибка компиляции паттерна '{pattern_str}': {exc}")
            
            if compiled_for_topic:
                self._compiled_patterns[topic.category] = compiled_for_topic
    
    def check_query(self, query: str) -> SafetyCheckResult:
        """
        Проверить запрос на наличие чувствительных тем.
        
        Args:
            query: Текст запроса пользователя
        
        Returns:
            SafetyCheckResult с результатами проверки
        """
        detected_categories = []
        matched_patterns = []
        
        query_lower = query.lower()
        
        for category, compiled_list in self._compiled_patterns.items():
            for pattern, topic in compiled_list:
                if pattern.search(query_lower):
                    if category not in detected_categories:
                        detected_categories.append(category)
                    matched_patterns.append(topic.description)
                    logger.debug(
                        f"Детектирована чувствительная тема [{category}] "
                        f"в запросе: {query[:50]}..."
                    )
        
        is_sensitive = len(detected_categories) > 0
        
        return SafetyCheckResult(
            is_sensitive=is_sensitive,
            detected_categories=detected_categories,
            matched_patterns=list(set(matched_patterns)),
            requires_human_review=is_sensitive,
        )
    
    def is_safe_query(self, query: str) -> bool:
        """
        Быстрая проверка: является ли запрос безопасным.
        
        Args:
            query: Текст запроса
        
        Returns:
            True если запрос не содержит чувствительных тем
        """
        result = self.check_query(query)
        return not result.is_sensitive


# ═══════════════════════════════════════════════════════════════
# Глобальный экземпляр детектора
# ═══════════════════════════════════════════════════════════════

_global_detector: Optional[SafetyDetector] = None


def get_safety_detector() -> SafetyDetector:
    """Получить глобальный экземпляр SafetyDetector."""
    global _global_detector
    if _global_detector is None:
        _global_detector = SafetyDetector()
    return _global_detector


def check_sensitivity(query: str) -> SafetyCheckResult:
    """
    Удобная функция для проверки запроса на чувствительность.
    
    Это обёртка над SafetyDetector.check_query().
    """
    detector = get_safety_detector()
    return detector.check_query(query)


def is_safe_query(query: str) -> bool:
    """
    Быстрая проверка безопасности запроса.
    
    Это обёртка над SafetyDetector.is_safe_query().
    """
    detector = get_safety_detector()
    return detector.is_safe_query(query)
