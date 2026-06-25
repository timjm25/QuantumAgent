"""
Base agent class for all QuantumAgent specialized agents.

Every agent inherits from BaseAgent which provides:
  - Unique identity and state machine
  - Structured task execution with retries
  - Memory read/write integration
  - Explainability attachment
  - Event publishing
  - Audit logging
"""
from __future__ import annotations

import asyncio
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.events import EventBus, EventType, Event, event_bus
from ..core.logging import get_logger


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentResult:
    success: bool
    output: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""
    supporting_refs: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    agent_id: str = ""
    task_id: str = ""
    duration_ms: float = 0.0


class BaseAgent:
    """
    Abstract base class for all QuantumAgent platform agents.
    Subclasses implement `_execute` with their domain logic.
    """

    name: str = "base_agent"
    description: str = "Base agent"
    capabilities: List[str] = []

    def __init__(
        self,
        agent_id: Optional[str] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        self.agent_id = agent_id or f"{self.name}_{uuid.uuid4().hex[:8]}"
        self.status = AgentStatus.IDLE
        self.bus = bus or event_bus
        self.logger = get_logger(self.name)
        self._task_history: List[AgentTask] = []
        self._memory: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    async def run(self, task_input: Dict[str, Any], task_name: str = "") -> AgentResult:
        task = AgentTask(name=task_name or self.name, input=task_input)
        self.status = AgentStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)

        await self._emit(EventType.AGENT_STARTED, {"agent": self.agent_id, "task": task.task_id})

        t0 = asyncio.get_event_loop().time()
        last_exc: Optional[Exception] = None

        while task.retry_count <= task.max_retries:
            try:
                result = await self._execute(task_input)
                result.agent_id = self.agent_id
                result.task_id = task.task_id
                result.duration_ms = (asyncio.get_event_loop().time() - t0) * 1000

                task.output = result.output
                task.completed_at = datetime.now(timezone.utc)
                self.status = AgentStatus.COMPLETED
                self._task_history.append(task)

                await self._emit(
                    EventType.AGENT_COMPLETED,
                    {
                        "agent": self.agent_id,
                        "task": task.task_id,
                        "duration_ms": result.duration_ms,
                        "confidence": result.confidence,
                    },
                )
                return result

            except Exception as exc:
                last_exc = exc
                task.retry_count += 1
                self.logger.warning(
                    "agent_retry",
                    agent=self.agent_id,
                    attempt=task.retry_count,
                    error=str(exc),
                )
                if task.retry_count <= task.max_retries:
                    await asyncio.sleep(2 ** task.retry_count)

        # All retries exhausted
        task.error = str(last_exc)
        task.completed_at = datetime.now(timezone.utc)
        self.status = AgentStatus.FAILED
        self._task_history.append(task)

        await self._emit(
            EventType.AGENT_FAILED,
            {"agent": self.agent_id, "task": task.task_id, "error": str(last_exc)},
        )

        return AgentResult(
            success=False,
            output={},
            reasoning=f"Agent failed after {task.max_retries} retries: {last_exc}",
            agent_id=self.agent_id,
            task_id=task.task_id,
            duration_ms=(asyncio.get_event_loop().time() - t0) * 1000,
        )

    def remember(self, key: str, value: Any) -> None:
        self._memory[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self._memory.get(key, default)

    def task_history(self) -> List[AgentTask]:
        return list(self._task_history)

    def status_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "tasks_completed": sum(1 for t in self._task_history if t.output is not None),
            "tasks_failed": sum(1 for t in self._task_history if t.error is not None),
        }

    # ------------------------------------------------------------------ #
    #  Subclass interface                                                  #
    # ------------------------------------------------------------------ #

    async def _execute(self, task_input: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError(f"{self.__class__.__name__} must implement _execute()")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    async def _emit(self, event_type: EventType, payload: Dict[str, Any]) -> None:
        try:
            await self.bus.publish(Event(type=event_type, payload=payload, source=self.agent_id))
        except Exception:
            pass
