# -*- coding: utf-8 -*-
"""
Koib-V-4.2 — Модуль генерации ответов (обновлённый)
=====================================================
Генерация с жёсткой привязкой к контексту, цитированием,
поддержкой различных LLM-провайдеров и пост-генерационной валидацией.

Поддерживаемые провайдеры:
  - GigaChat (Сбер)
  - OpenAI (GPT-4o-mini и др.)
  - Локальная LLM (Ollama / llama-cpp)
"""

import logging
from typing import List, Dict, Any, Optional

from .retrieval import RetrievalResult
from config import (
    LLM_PROVIDER,
    GIGACHAT_CREDENTIALS, GIGACHAT_MODEL, GIGACHAT_TEMPERATURE, GIGACHAT_MAX_TOKENS,
    OPENAI_API_KEY, OPENAI_LLM_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS,
    LOCAL_LLM_MODEL, LOCAL_LLM_URL,
)

logger = logging.getLogger("koib.generation")


# ═══════════════════════════════════════════════════════════════
# Системный промпт с жёсткой привязкой к контексту
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Ты — эксперт-ассистент по технической документации. Твоя задача — отвечать на вопросы пользователя строго на основе предоставленного контекста из документации.

ПРАВИЛА ОТВЕТА:

1. **Опирайся ТОЛЬКО на предоставленный контекст.** Не придумывай информацию, которой нет в контекстных фрагментах. Если контекст не содержит ответа — честно сообщи: «В предоставленной документации нет информации по этому вопросу.»

2. **Цитируй источники.** Каждое утверждение в ответе должно сопровождаться ссылкой на источник в формате: [Документ: {имя_файла}, стр. {номер}]. Если информация из нескольких источников — укажи все.

3. **Таблицы.** Если в контексте есть таблица и она релевантна вопросу, воспроизведи её в формате Markdown, затем прокомментируй данные.

4. **Формулы.** Если в контексте есть формулы, выведи их в формате LaTeX и объясни значение переменных. Если переменные не объяснены в контексте — укажи это.

5. **Схемы и рисунки.** Если контекст содержит описание рисунка или схемы, опиши его текстуально и укажи источник.

6. **Структура ответа.** Отвечай структурированно: используй заголовки, списки, выделение важного. Начинай с прямого ответа, затем давай пояснения.

7. **Не повторяй вопрос.** Переходи сразу к ответу.

8. **Язык.** Отвечай на том же языке, на котором задан вопрос (по умолчанию — русский)."""


# ═══════════════════════════════════════════════════════════════
# Формирование промпта с контекстом
# ═══════════════════════════════════════════════════════════════

def build_prompt(query: str, results: List[RetrievalResult]) -> str:
    """
    Сформировать промпт для LLM с контекстом из результатов поиска.

    Формат контекста:
      [Документ: имя_файла, стр. X]
      Раздел: Заголовок
      Содержимое фрагмента

    Для таблиц и формул подставляется полный контент из docstore.
    """
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"--- Фрагмент {i} ---")
        context_parts.append(r.to_context_string())
        context_parts.append("")

    context_text = "\n".join(context_parts)

    prompt = f"""КОНТЕКСТ ИЗ ТЕХНИЧЕСКОЙ ДОКУМЕНТАЦИИ:
{context_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Ответь на вопрос, строго опираясь на приведённый выше контекст. Обязательно цитируй источники в формате [Документ: имя_файла, стр. X]."""

    return prompt


# ═══════════════════════════════════════════════════════════════
# LLM-клиенты
# ═══════════════════════════════════════════════════════════════

class LLMClient:
    """Унифицированный клиент для разных LLM-провайдеров."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or LLM_PROVIDER
        self._client = None

    def _init_gigachat(self):
        """Инициализация GigaChat-клиента."""
        try:
            import sys
            from pathlib import Path
            root = Path(__file__).parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

            from gigachat_client import GigaChatClient
            creds = GIGACHAT_CREDENTIALS
            if not creds:
                raise ValueError("GIGACHAT_CREDENTIALS не задан")
            self._client = GigaChatClient(creds)
            return self._client
        except ImportError:
            # Реализуем встроенный клиент
            return _GigaChatDirect()

    def _init_openai(self):
        """Инициализация OpenAI-клиента."""
        try:
            from openai import OpenAI
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY не задан")
            self._client = OpenAI(api_key=OPENAI_API_KEY)
            return self._client
        except ImportError:
            raise ImportError("openai не установлен: pip install openai")

    def _init_local(self):
        """Инициализация локальной LLM (Ollama)."""
        try:
            import requests as req
            # Проверяем доступность Ollama
            resp = req.get(f"{LOCAL_LLM_URL}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama недоступна: {LOCAL_LLM_URL}")
            self._client = "ollama"
            return self._client
        except Exception as exc:
            raise ConnectionError(f"Локальная LLM недоступна: {exc}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        """
        Сгенерировать ответ LLM.

        Args:
            prompt:          Пользовательский промпт
            system_prompt:   Системный промпт (по умолчанию — наш)
            max_tokens:      Максимум токенов в ответе
            temperature:     Температура генерации

        Returns:
            Текст ответа
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT

        if self.provider == "gigachat":
            return self._generate_gigachat(prompt, sys_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._generate_openai(prompt, sys_prompt, max_tokens, temperature)
        elif self.provider == "local":
            return self._generate_local(prompt, sys_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Неизвестный LLM-провайдер: {self.provider}")

    def _generate_gigachat(self, prompt: str, system_prompt: str,
                           max_tokens: int, temperature: float) -> str:
        """Генерация через GigaChat."""
        if self._client is None:
            self._client = self._init_gigachat()

        try:
            # GigaChatClient (внешний модуль)
            if hasattr(self._client, 'chat'):
                full_prompt = f"{system_prompt}\n\n{prompt}"
                return self._client.chat(full_prompt, temperature=temperature, max_tokens=max_tokens)
            # Встроенный клиент
            elif hasattr(self._client, 'ask'):
                return self._client.ask(prompt, system_prompt, max_tokens, temperature)
            else:
                return self._generate_gigachat_direct(prompt, system_prompt, max_tokens, temperature)
        except Exception as exc:
            logger.error(f"Ошибка GigaChat: {exc}")
            return f"Ошибка генерации ответа: {exc}"

    def _generate_gigachat_direct(self, prompt: str, system_prompt: str,
                                   max_tokens: int, temperature: float) -> str:
        """Прямой вызов GigaChat API (без внешнего клиента)."""
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        creds = GIGACHAT_CREDENTIALS
        if not creds:
            return "Ошибка: GIGACHAT_CREDENTIALS не задан"

        # Получаем токен
        auth_resp = requests.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={
                "Authorization": f"Basic {creds}",
                "RqUID": "koib-rag-001",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": "GIGACHAT_API_PERS"},
            verify=False,
            timeout=30,
        )

        if auth_resp.status_code != 200:
            return f"Ошибка авторизации GigaChat: {auth_resp.status_code}"

        token = auth_resp.json()["access_token"]

        # Генерация
        chat_resp = requests.post(
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": GIGACHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            verify=False,
            timeout=120,
        )

        if chat_resp.status_code != 200:
            return f"Ошибка GigaChat API: {chat_resp.status_code}"

        return chat_resp.json()["choices"][0]["message"]["content"].strip()

    def _generate_openai(self, prompt: str, system_prompt: str,
                         max_tokens: int, temperature: float) -> str:
        """Генерация через OpenAI API."""
        if self._client is None:
            self._client = self._init_openai()

        try:
            response = self._client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"Ошибка OpenAI: {exc}")
            return f"Ошибка генерации ответа: {exc}"

    def _generate_local(self, prompt: str, system_prompt: str,
                        max_tokens: int, temperature: float) -> str:
        """Генерация через локальную LLM (Ollama)."""
        if self._client is None:
            self._client = self._init_local()

        try:
            import requests as req
            response = req.post(
                f"{LOCAL_LLM_URL}/api/generate",
                json={
                    "model": LOCAL_LLM_MODEL.split("/")[-1],  # Ollama использует короткие имена
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=120,
            )
            return response.json().get("response", "").strip()
        except Exception as exc:
            logger.error(f"Ошибка локальной LLM: {exc}")
            return f"Ошибка генерации ответа: {exc}"


class _GigaChatDirect:
    """Встроенный GigaChat-клиент (заглушка для инициализации)."""
    pass


# ═══════════════════════════════════════════════════════════════
# Основная функция генерации
# ═══════════════════════════════════════════════════════════════

class AnswerGenerator:
    """
    Генератор ответов на основе RAG.

    Пайплайн (обновлённый):
      1. Поиск релевантных фрагментов (HybridRetriever) с фильтрацией карантина
      2. Детекция чувствительных тем (safety.py)
      3. Формирование промпта с контекстом
      4. Генерация ответа через LLM
      5. Пост-генерационная валидация (validation.py)
      6. Логирование запроса (logging_module.py)
    """

    def __init__(self, retriever=None, llm_client: Optional[LLMClient] = None):
        from .retrieval import HybridRetriever
        self.retriever = retriever or HybridRetriever()
        self.llm = llm_client or LLMClient()
        
        # Инициализируем модули валидации и безопасности
        self._validator = None
        self._safety_detector = None
        self._query_logger = None
    
    def _get_validator(self):
        """Ленивая инициализация валидатора."""
        if self._validator is None:
            try:
                from .validation import AnswerValidator
                # Переиспользуем эмбеддинги из retriever
                embeddings = self.retriever.index_builder.embeddings
                self._validator = AnswerValidator(embeddings=embeddings)
            except Exception as exc:
                logger.warning(f"Ошибка инициализации валидатора: {exc}")
                self._validator = None
        return self._validator
    
    def _get_safety_detector(self):
        """Ленивая инициализация детектора безопасности."""
        if self._safety_detector is None:
            try:
                from .safety import get_safety_detector
                self._safety_detector = get_safety_detector()
            except Exception as exc:
                logger.warning(f"Ошибка инициализации детектора безопасности: {exc}")
                self._safety_detector = None
        return self._safety_detector
    
    def _get_query_logger(self):
        """Ленивая инициализация логгера."""
        if self._query_logger is None:
            try:
                from .logging_module import get_query_logger
                self._query_logger = get_query_logger()
            except Exception as exc:
                logger.warning(f"Ошибка инициализации логгера: {exc}")
                self._query_logger = None
        return self._query_logger

    def answer(
        self,
        query: str,
        k: int = 5,
        model_filter: str = "",
        use_hyde: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Полный RAG-ответ: поиск + генерация + валидация.

        Returns:
            {
                "answer": str,           # Текст ответа
                "sources": List[Dict],   # Источники с цитатами
                "results": List[RetrievalResult],  # Найденные фрагменты
                "context_text": str,     # Полный контекст для LLM
                "validation": Dict,      # Результат валидации
                "status": str,           # approved | review | rejected
            }
        """
        # ── 1. Поиск ────────────────────────────────────────────
        results = self.retriever.search(query, k=k, model_filter=model_filter, use_hyde=use_hyde)

        if not results:
            result = {
                "answer": "В предоставленной документации нет информации по данному вопросу.",
                "sources": [],
                "results": [],
                "context_text": "",
                "validation": {"status": "review", "reason": "no_context"},
                "status": "review",
            }
            # Логируем
            self._log_query(query, result, model_filter)
            return result

        # ── 2. Детекция чувствительных тем ─────────────────────
        safety_result = None
        safety_detector = self._get_safety_detector()
        if safety_detector:
            try:
                safety_result = safety_detector.check_query(query)
            except Exception as exc:
                logger.warning(f"Ошибка проверки безопасности: {exc}")

        # ── 3. Формирование промпта ─────────────────────────────
        context_text = build_prompt(query, results)

        # ── 4. Генерация ────────────────────────────────────────
        answer = self.llm.generate(context_text)

        # ── 5. Извлекаем источники ──────────────────────────────
        sources = []
        seen_sources = set()
        for r in results:
            key = f"{r.source}:{r.page}"
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({
                    "document": r.source,
                    "page": r.page,
                    "heading": r.heading,
                    "chunk_type": r.chunk_type,
                    "score": r.score,
                })

        # ── 6. Пост-генерационная валидация ─────────────────────
        validator = self._get_validator()
        validation_result = None
        
        if validator:
            try:
                validation_result = validator.validate(
                    answer=answer,
                    context_chunks=results,
                    query=query,
                )
            except Exception as exc:
                logger.warning(f"Ошибка валидации ответа: {exc}")

        # ── 7. Определяем итоговый статус ───────────────────────
        status = "approved"
        final_answer = answer
        
        # Проверяем валидацию
        if validation_result:
            if validation_result.status == "rejected":
                # Маркеры неопределённости — блокируем ответ
                status = "rejected"
                final_answer = self._get_blocked_response()
            elif validation_result.status == "review":
                status = "review"
        
        # Проверяем чувствительные темы
        if safety_result and safety_result.is_sensitive:
            if status != "rejected":
                status = "review"

        # ── 8. Логируем запрос ──────────────────────────────────
        result = {
            "answer": final_answer,
            "sources": sources,
            "results": results,
            "context_text": context_text,
            "validation": validation_result.to_dict() if validation_result else {},
            "status": status,
            "safety": safety_result.to_dict() if safety_result else {},
        }
        
        self._log_query(query, result, model_filter)

        return result
    
    def _get_blocked_response(self) -> str:
        """Вернуть шаблонный ответ при блокировке."""
        try:
            from .validation import get_blocked_response
            return get_blocked_response()
        except:
            return "По вашему запросу не найдено точного ответа в официальных источниках."
    
    def _log_query(self, query: str, result: Dict[str, Any], model_filter: str) -> None:
        """Записать запрос в лог."""
        logger_instance = self._get_query_logger()
        if logger_instance:
            try:
                logger_instance.log(
                    query=query,
                    answer=result.get("answer", ""),
                    model_type=model_filter,
                    sources=result.get("sources", []),
                    validation_result=result.get("validation", {}),
                    status=result.get("status", "unknown"),
                    extra_metadata={
                        "safety": result.get("safety", {}),
                        "num_chunks": len(result.get("results", [])),
                    },
                )
            except Exception as exc:
                logger.warning(f"Ошибка логирования: {exc}")
