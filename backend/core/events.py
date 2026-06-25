"""
Platform event bus — lightweight pub/sub used by agents to coordinate
without direct coupling. Backed by asyncio queues in development;
swap the backend for RabbitMQ / Kafka in production.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    EXPERIMENT_SUBMITTED = "experiment.submitted"
    EXPERIMENT_COMPLETED = "experiment.completed"
    LITERATURE_FOUND = "literature.found"
    CIRCUIT_OPTIMISED = "circuit.optimised"
    MODEL_UPDATED = "model.updated"
    KNOWLEDGE_GRAPH_UPDATED = "kg.updated"
    HUMAN_APPROVAL_REQUIRED = "approval.required"
    HUMAN_APPROVAL_GRANTED = "approval.granted"
    HARDWARE_JOB_SUBMITTED = "hardware.submitted"
    HARDWARE_JOB_COMPLETED = "hardware.completed"


@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any]
    source: str = "platform"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None


class EventBus:
    """In-process event bus with asyncio queues."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    async def dispatch(self) -> None:
        """Drain the queue and invoke handlers. Run as a background task."""
        while True:
            event = await self._queue.get()
            handlers = self._subscribers.get(event.type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as exc:
                    pass
            self._queue.task_done()


event_bus = EventBus()
