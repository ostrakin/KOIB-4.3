# -*- coding: utf-8 -*-
"""
Koib-V-4.2 — Модуль карантина чанков
======================================
Механизм изоляции проблемных чанков:
  - Глобальный список quarantined_chunk_ids
  - Проверка при поиске
  - Функции добавления/удаления чанков из карантина
"""

import json
import logging
from pathlib import Path
from typing import Set, Optional, List
from datetime import datetime

logger = logging.getLogger("koib.quarantine")


class ChunkQuarantine:
    """
    Менеджер карантина для проблемных чанков.
    
    Чанки в карантине исключаются из результатов поиска.
    Список сохраняется в файл для персистентности между запусками.
    """
    
    def __init__(self, quarantine_file: Optional[Path] = None):
        """
        Args:
            quarantine_file: Путь к файлу хранения списка карантина
        """
        self.quarantine_file = quarantine_file or Path(
            __file__
        ).parent.parent / "output" / "metadata" / "quarantined_chunks.json"
        
        self._quarantined_ids: Set[str] = set()
        self._quarantine_metadata: dict = {}  # chunk_id -> {reason, timestamp, ...}
        
        self._load()
    
    def _load(self) -> None:
        """Загрузить список карантина из файла."""
        if not self.quarantine_file.exists():
            logger.info("Файл карантина не найден, создан пустой список")
            return
        
        try:
            with open(self.quarantine_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._quarantined_ids = set(data.get("chunk_ids", []))
            self._quarantine_metadata = data.get("metadata", {})
            
            logger.info(
                f"Загружен карантин: {len(self._quarantined_ids)} чанков"
            )
        except Exception as exc:
            logger.warning(f"Ошибка загрузки карантина: {exc}")
            self._quarantined_ids = set()
            self._quarantine_metadata = {}
    
    def _save(self) -> None:
        """Сохранить список карантина в файл."""
        try:
            self.quarantine_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "chunk_ids": list(self._quarantined_ids),
                "metadata": self._quarantine_metadata,
                "last_updated": datetime.now().isoformat(),
            }
            
            with open(self.quarantine_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Карантин сохранён: {len(self._quarantined_ids)} чанков")
        except Exception as exc:
            logger.error(f"Ошибка сохранения карантина: {exc}")
    
    def add(
        self,
        chunk_id: str,
        reason: str = "",
        auto_expire_hours: Optional[int] = None,
    ) -> bool:
        """
        Добавить чанк в карантин.
        
        Args:
            chunk_id: Идентификатор чанка
            reason: Причина помещения в карантин
            auto_expire_hours: Автоматическое удаление через N часов (опционально)
        
        Returns:
            True если чанк добавлен, False если уже был в карантине
        """
        if chunk_id in self._quarantined_ids:
            logger.debug(f"Чанк {chunk_id} уже в карантине")
            return False
        
        self._quarantined_ids.add(chunk_id)
        self._quarantine_metadata[chunk_id] = {
            "reason": reason,
            "added_at": datetime.now().isoformat(),
            "auto_expire_hours": auto_expire_hours,
        }
        
        self._save()
        logger.info(f"Чанк {chunk_id} помещён в карантин: {reason}")
        return True
    
    def remove(self, chunk_id: str) -> bool:
        """
        Удалить чанк из карантина.
        
        Args:
            chunk_id: Идентификатор чанка
        
        Returns:
            True если чанк удалён, False если не был в карантине
        """
        if chunk_id not in self._quarantined_ids:
            logger.debug(f"Чанк {chunk_id} не в карантине")
            return False
        
        self._quarantined_ids.discard(chunk_id)
        self._quarantine_metadata.pop(chunk_id, None)
        
        self._save()
        logger.info(f"Чанк {chunk_id} удалён из карантина")
        return True
    
    def is_quarantined(self, chunk_id: str) -> bool:
        """
        Проверить, находится ли чанк в карантине.
        
        Args:
            chunk_id: Идентификатор чанка
        
        Returns:
            True если чанк в карантине
        """
        # Проверяем автоматическое истечение срока
        if chunk_id in self._quarantine_metadata:
            meta = self._quarantine_metadata[chunk_id]
            if meta.get("auto_expire_hours"):
                added_at = datetime.fromisoformat(meta["added_at"])
                elapsed_hours = (datetime.now() - added_at).total_seconds() / 3600
                
                if elapsed_hours >= meta["auto_expire_hours"]:
                    logger.info(
                        f"Истёк срок карантина для чанка {chunk_id}, "
                        f"автоматическое удаление"
                    )
                    self.remove(chunk_id)
                    return False
        
        return chunk_id in self._quarantined_ids
    
    def filter_chunks(self, chunks: List) -> List:
        """
        Отфильтровать чанки, исключив находящиеся в карантине.
        
        Args:
            chunks: Список объектов с атрибутом chunk_id
        
        Returns:
            Отфильтрованный список
        """
        filtered = []
        removed_count = 0
        
        for chunk in chunks:
            chunk_id = getattr(chunk, 'chunk_id', None)
            if chunk_id and self.is_quarantined(chunk_id):
                removed_count += 1
                logger.debug(f"Исключён чанк из карантина: {chunk_id}")
            else:
                filtered.append(chunk)
        
        if removed_count > 0:
            logger.info(f"Исключено {removed_count} чанков из карантина")
        
        return filtered
    
    def get_all_quarantined(self) -> List[dict]:
        """
        Получить список всех чанков в карантине с метаданными.
        
        Returns:
            Список словарей {chunk_id, reason, added_at, ...}
        """
        result = []
        for chunk_id in self._quarantined_ids:
            meta = self._quarantine_metadata.get(chunk_id, {})
            result.append({
                "chunk_id": chunk_id,
                **meta,
            })
        return result
    
    def clear(self) -> int:
        """
        Очистить весь карантин.
        
        Returns:
            Количество удалённых записей
        """
        count = len(self._quarantined_ids)
        self._quarantined_ids.clear()
        self._quarantine_metadata.clear()
        self._save()
        logger.info(f"Карантин очищен: удалено {count} записей")
        return count
    
    @property
    def size(self) -> int:
        """Количество чанков в карантине."""
        return len(self._quarantined_ids)


# ═══════════════════════════════════════════════════════════════
# Глобальный экземпляр карантина
# ═══════════════════════════════════════════════════════════════

_global_quarantine: Optional[ChunkQuarantine] = None


def get_quarantine_manager() -> ChunkQuarantine:
    """Получить глобальный экземпляр ChunkQuarantine."""
    global _global_quarantine
    if _global_quarantine is None:
        _global_quarantine = ChunkQuarantine()
    return _global_quarantine


def quarantine_chunk(
    chunk_id: str,
    reason: str = "",
    auto_expire_hours: Optional[int] = None,
) -> bool:
    """
    Удобная функция для помещения чанка в карантин.
    
    Это обёртка над ChunkQuarantine.add().
    """
    manager = get_quarantine_manager()
    return manager.add(chunk_id, reason=reason, auto_expire_hours=auto_expire_hours)


def unquarantine_chunk(chunk_id: str) -> bool:
    """
    Удобная функция для удаления чанка из карантина.
    
    Это обёртка над ChunkQuarantine.remove().
    """
    manager = get_quarantine_manager()
    return manager.remove(chunk_id)


def is_chunk_quarantined(chunk_id: str) -> bool:
    """
    Удобная функция для проверки статуса карантина.
    
    Это обёртка над ChunkQuarantine.is_quarantined().
    """
    manager = get_quarantine_manager()
    return manager.is_quarantined(chunk_id)


def filter_quarantined_chunks(chunks: List) -> List:
    """
    Удобная функция для фильтрации чанков.
    
    Это обёртка над ChunkQuarantine.filter_chunks().
    """
    manager = get_quarantine_manager()
    return manager.filter_chunks(chunks)
