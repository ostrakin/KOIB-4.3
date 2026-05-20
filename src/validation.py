# -*- coding: utf-8 -*-
"""
Koib-V-4.2 — Модуль валидации ответов
========================================
Пост-генерационная валидация ответов LLM:
  1. Проверка наличия ссылок на источники
  2. Детекция маркеров неопределённости
  3. Семантическая проверка согласованности с контекстом
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("koib.validation")


# ═══════════════════════════════════════════════════════════════
# Маркеры неопределённости (недопустимы в ответах)
# ═══════════════════════════════════════════════════════════════

UNCERTAINTY_MARKERS = [
    # Русские фразы
    "возможно",
    "вероятно",
    "предполагается",
    "скорее всего",
    "может быть",
    "по-видимому",
    "очевидно",
    "кажется",
    "вероятнее всего",
    "вполне возможно",
    "не исключено",
    "можно предположить",
    "судя по всему",
    "по всей видимости",
    "наверное",
    "вроде бы",
    "как будто",
    "примерно",
    "ориентировочно",
    # Английские (на случай смешения)
    "maybe",
    "perhaps",
    "probably",
    "likely",
    "possibly",
    "it seems",
    "apparently",
    "presumably",
]

# Регулярные выражения для более точного поиска
UNCERTAINTY_PATTERNS = [
    r"\bвозможно\b",
    r"\bвероятно\b",
    r"\bпредполагает(?:ся|ся)\b",
    r"\bскорее всего\b",
    r"\bможет быть\b",
    r"\bпо-видимому\b",
    r"\bочевидно\b",
    r"\bкажется\b",
    r"\bвероятнее всего\b",
    r"\bвполне возможно\b",
    r"\bне исключено\b",
    r"\bможно предположить\b",
    r"\bсудя по всему\b",
    r"\bпо всей видимости\b",
    r"\bнаверное\b",
    r"\bвроде бы\b",
    r"\bкак будто\b",
    r"\bпримерно\b",
    r"\bориентировочно\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bpossibly\b",
    r"\bit seems\b",
    r"\bapparently\b",
    r"\bpresumably\b",
]


# ═══════════════════════════════════════════════════════════════
# Паттерны для детекции ссылок на источники
# ═══════════════════════════════════════════════════════════════

SOURCE_PATTERNS = [
    # [Документ: имя_файла, стр. X]
    r"\[Документ:\s*[^,\]]+,\s*стр\.\s*\d+\]",
    # [Документ: имя_файла, страница X]
    r"\[Документ:\s*[^,\]]+,\s*страница\s*\d+\]",
    # (источник: ...)
    r"\(источник:\s*[^)]+\)",
    # см. раздел ...
    r"\bсм\.\s+раздел\s+",
    # согласно документации
    r"\bсогласно\s+(?:технической\s+)?документации\b",
]


# ═══════════════════════════════════════════════════════════════
# Результаты валидации
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationCheck:
    """Результат одной проверки."""
    name: str
    passed: bool
    details: str = ""
    severity: str = "info"  # info | warning | critical


@dataclass
class ValidationResult:
    """Общий результат валидации ответа."""
    status: str = "approved"  # approved | review | rejected
    checks: List[ValidationCheck] = field(default_factory=list)
    requires_review_reasons: List[str] = field(default_factory=list)
    uncertainty_found: Optional[str] = None
    sources_found: List[str] = field(default_factory=list)
    semantic_similarity: float = 1.0
    
    def add_check(self, check: ValidationCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            if check.severity == "critical":
                self.status = "rejected"
            elif check.severity == "warning" and self.status != "rejected":
                if self.status == "approved":
                    self.status = "review"
                self.requires_review_reasons.append(check.details)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализовать в словарь для логирования."""
        return {
            "status": self.status,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "details": c.details,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
            "requires_review_reasons": self.requires_review_reasons,
            "uncertainty_found": self.uncertainty_found,
            "sources_found": self.sources_found,
            "semantic_similarity": self.semantic_similarity,
        }


# ═══════════════════════════════════════════════════════════════
# Валидатор ответов
# ═══════════════════════════════════════════════════════════════

class AnswerValidator:
    """
    Валидатор ответов LLM.
    
    Выполняет проверки:
      1. Наличие ссылок на источники
      2. Отсутствие маркеров неопределённости
      3. Семантическая согласованность с контекстом
    """
    
    def __init__(self, embeddings=None, similarity_threshold: float = 0.75):
        """
        Args:
            embeddings: Объект эмбеддингов (для семантической проверки)
            similarity_threshold: Порог косинусного сходства (по умолчанию 0.75)
        """
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
    
    def validate(
        self,
        answer: str,
        context_chunks: List[Any],
        query: str = "",
    ) -> ValidationResult:
        """
        Выполнить полную валидацию ответа.
        
        Args:
            answer: Текст ответа от LLM
            context_chunks: Список RetrievalResult (контекстные чанки)
            query: Исходный запрос пользователя (опционально)
        
        Returns:
            ValidationResult со статусом и деталями проверок
        """
        result = ValidationResult()
        
        # 1. Проверка на маркеры неопределённости (критическая)
        uncertainty_check = self._check_uncertainty(answer)
        result.add_check(uncertainty_check)
        
        # Если найдены маркеры неопределённости — сразу rejected
        if not uncertainty_check.passed:
            result.uncertainty_found = uncertainty_check.details
            return result
        
        # 2. Проверка наличия ссылок на источники
        sources_check = self._check_sources(answer)
        result.add_check(sources_check)
        result.sources_found = sources_check.details.split("; ") if sources_check.passed else []
        
        # 3. Семантическая проверка (если есть embeddings)
        if self.embeddings and context_chunks:
            semantic_check = self._check_semantic_consistency(answer, context_chunks)
            result.add_check(semantic_check)
            result.semantic_similarity = semantic_check.semantic_similarity
        
        return result
    
    def _check_uncertainty(self, answer: str) -> ValidationCheck:
        """Проверить наличие маркеров неопределённости."""
        answer_lower = answer.lower()
        
        found_markers = []
        for pattern in UNCERTAINTY_PATTERNS:
            matches = re.findall(pattern, answer_lower, re.IGNORECASE)
            if matches:
                found_markers.extend(matches)
        
        if found_markers:
            unique_markers = list(set(found_markers))[:3]  # Первые 3 уникальных
            return ValidationCheck(
                name="uncertainty_check",
                passed=False,
                details=f"Найдены маркеры неопределённости: {', '.join(unique_markers)}",
                severity="critical",
            )
        
        return ValidationCheck(
            name="uncertainty_check",
            passed=True,
            details="Маркеры неопределённости не найдены",
            severity="info",
        )
    
    def _check_sources(self, answer: str) -> ValidationCheck:
        """Проверить наличие ссылок на источники."""
        found_sources = []
        
        for pattern in SOURCE_PATTERNS:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            found_sources.extend(matches)
        
        if found_sources:
            unique_sources = list(set(found_sources))
            return ValidationCheck(
                name="sources_check",
                passed=True,
                details="; ".join(unique_sources[:5]),  # Первые 5 уникальных
                severity="info",
            )
        
        return ValidationCheck(
            name="sources_check",
            passed=False,
            details="Ссылки на источники не найдены",
            severity="warning",
        )
    
    def _check_semantic_consistency(
        self,
        answer: str,
        context_chunks: List[Any],
    ) -> ValidationCheck:
        """
        Семантическая проверка: сравнение эмбеддинга ответа 
        с эмбеддингами контекстных чанков.
        """
        try:
            # Получаем эмбеддинг ответа
            answer_embedding = self._get_embedding(answer)
            if answer_embedding is None:
                return ValidationCheck(
                    name="semantic_check",
                    passed=False,
                    details="Не удалось получить эмбеддинг ответа",
                    severity="warning",
                )
            
            # Получаем эмбеддинги контекстов
            context_texts = []
            for chunk in context_chunks[:5]:  # Берём топ-5 чанков
                if hasattr(chunk, 'content'):
                    context_texts.append(chunk.content)
                elif isinstance(chunk, dict):
                    context_texts.append(chunk.get('content', ''))
            
            if not context_texts:
                return ValidationCheck(
                    name="semantic_check",
                    passed=False,
                    details="Контекстные чанки пусты",
                    severity="warning",
                )
            
            # Вычисляем косинусное сходство
            max_similarity = 0.0
            for ctx_text in context_texts:
                ctx_embedding = self._get_embedding(ctx_text)
                if ctx_embedding is not None:
                    similarity = self._cosine_similarity(answer_embedding, ctx_embedding)
                    max_similarity = max(max_similarity, similarity)
            
            passed = max_similarity >= self.similarity_threshold
            
            return ValidationCheck(
                name="semantic_check",
                passed=passed,
                details=f"Косинусное сходство: {max_similarity:.3f} (порог: {self.similarity_threshold})",
                severity="warning",
                semantic_similarity=max_similarity,
            )
            
        except Exception as exc:
            logger.warning(f"Ошибка семантической проверки: {exc}")
            return ValidationCheck(
                name="semantic_check",
                passed=False,
                details=f"Ошибка проверки: {exc}",
                severity="info",
            )
    
    def _get_embedding(self, text: str) -> Optional[list]:
        """Получить эмбеддинг текста."""
        try:
            if hasattr(self.embeddings, 'embed_query'):
                return self.embeddings.embed_query(text)
            elif hasattr(self.embeddings, 'embed_documents'):
                return self.embeddings.embed_documents([text])[0]
            else:
                logger.warning("Эмбеддинги не поддерживают нужный интерфейс")
                return None
        except Exception as exc:
            logger.warning(f"Ошибка получения эмбеддинга: {exc}")
            return None
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Вычислить косинусное сходство двух векторов."""
        import numpy as np
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))


# ═══════════════════════════════════════════════════════════════
# Шаблонный ответ при блокировке
# ═══════════════════════════════════════════════════════════════

BLOCKED_RESPONSE_TEMPLATE = (
    "По вашему запросу не найдено точного ответа в официальных источниках."
)


def get_blocked_response() -> str:
    """Вернуть шаблонный ответ при блокировке."""
    return BLOCKED_RESPONSE_TEMPLATE
