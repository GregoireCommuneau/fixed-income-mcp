"""
models/alerts.py
Alert system — Strategy Pattern for conditions, Observer for dispatch.

Architecture:
  AlertCondition (ABC)
    ├── ThresholdCondition     — value > / < / == threshold
    ├── CrossoverCondition     — upward/downward crossover of a level
    ├── SpreadCondition        — spread between two series
    └── ZScoreCondition        — deviation from the historical mean

  Alert              — links a condition to an instrument/series + metadata
  AlertResult        — result of an evaluation (triggered or not)
  AlertManager       — registry + dispatcher (Observer)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class AlertSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    ACTIVE    = "active"
    TRIGGERED = "triggered"
    SNOOZED   = "snoozed"
    DISABLED  = "disabled"

class ComparisonOp(str, Enum):
    GT  = ">"
    GTE = ">="
    LT  = "<"
    LTE = "<="
    EQ  = "=="
    NEQ = "!="


# ──────────────────────────────────────────────
# AlertCondition — Strategy Pattern
# ──────────────────────────────────────────────

class AlertCondition(ABC):
    """Strategy interface for all alert conditions."""

    @abstractmethod
    def evaluate(self, value: float, context: dict[str, Any] | None = None) -> bool:
        """Returns True if the condition is triggered."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the condition."""
        ...


class ThresholdCondition(AlertCondition):
    """Triggers if value OP threshold. E.g.: spread_2s10s < -10 bps."""

    def __init__(self, op: ComparisonOp, threshold: float, unit: str = ""):
        self.op = op
        self.threshold = threshold
        self.unit = unit

        self._ops: dict[ComparisonOp, Callable[[float, float], bool]] = {
            ComparisonOp.GT:  lambda a, b: a > b,
            ComparisonOp.GTE: lambda a, b: a >= b,
            ComparisonOp.LT:  lambda a, b: a < b,
            ComparisonOp.LTE: lambda a, b: a <= b,
            ComparisonOp.EQ:  lambda a, b: a == b,
            ComparisonOp.NEQ: lambda a, b: a != b,
        }

    def evaluate(self, value: float, context: dict[str, Any] | None = None) -> bool:
        return self._ops[self.op](value, self.threshold)

    def describe(self) -> str:
        return f"value {self.op.value} {self.threshold} {self.unit}"


class CrossoverCondition(AlertCondition):
    """
    Triggers when the value crosses a level (up or down).
    Requires a context with 'previous_value'.
    """

    def __init__(self, level: float, direction: str = "both"):
        assert direction in ("up", "down", "both")
        self.level = level
        self.direction = direction

    def evaluate(self, value: float, context: dict[str, Any] | None = None) -> bool:
        if not context or "previous_value" not in context:
            return False
        prev = context["previous_value"]
        crossed_up   = prev < self.level <= value
        crossed_down = prev > self.level >= value
        if self.direction == "up":
            return crossed_up
        if self.direction == "down":
            return crossed_down
        return crossed_up or crossed_down

    def describe(self) -> str:
        return f"crosses {self.level} ({self.direction})"


class SpreadCondition(AlertCondition):
    """Triggers if the spread between two values exceeds a threshold (in bps)."""

    def __init__(self, op: ComparisonOp, threshold_bps: float):
        self.op = op
        self.threshold_bps = threshold_bps

    def evaluate(self, value: float, context: dict[str, Any] | None = None) -> bool:
        if not context or "second_value" not in context:
            return False
        spread_bps = (value - context["second_value"]) * 10_000
        ops = {
            ComparisonOp.GT:  spread_bps > self.threshold_bps,
            ComparisonOp.GTE: spread_bps >= self.threshold_bps,
            ComparisonOp.LT:  spread_bps < self.threshold_bps,
            ComparisonOp.LTE: spread_bps <= self.threshold_bps,
        }
        return ops.get(self.op, False)

    def describe(self) -> str:
        return f"spread {self.op.value} {self.threshold_bps} bps"


class ZScoreCondition(AlertCondition):
    """Triggers if the z-score (deviation/std) exceeds a threshold."""

    def __init__(self, threshold: float, op: ComparisonOp = ComparisonOp.GT):
        self.threshold = threshold
        self.op = op

    def evaluate(self, value: float, context: dict[str, Any] | None = None) -> bool:
        if not context or "mean" not in context or "std" not in context:
            return False
        std = context["std"]
        if std == 0:
            return False
        z = (value - context["mean"]) / std
        ops = {
            ComparisonOp.GT:  abs(z) > self.threshold,
            ComparisonOp.LT:  abs(z) < self.threshold,
            ComparisonOp.GTE: abs(z) >= self.threshold,
        }
        return ops.get(self.op, False)

    def describe(self) -> str:
        return f"|z-score| {self.op.value} {self.threshold}"


# ──────────────────────────────────────────────
# AlertResult
# ──────────────────────────────────────────────

class AlertResult(BaseModel):
    """Result of an alert evaluation."""

    alert_id: str
    alert_name: str
    triggered: bool
    value: float
    condition_desc: str
    severity: AlertSeverity
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str = ""
    context: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(self.severity, "•")
        status = "TRIGGERED" if self.triggered else "OK"
        return (
            f"{icon} [{status}] {self.alert_name} | "
            f"Value: {self.value:.4f} | Condition: {self.condition_desc} | "
            f"{self.timestamp.strftime('%H:%M:%S')}"
        )


# ──────────────────────────────────────────────
# Alert
# ──────────────────────────────────────────────

class Alert(BaseModel):
    """
    An alert = a condition + a target + metadata.
    Uses model_config to store the condition (non-Pydantic).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    series_id: str                      # ex: "DGS10", "2s10s_spread", "TLT"
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    last_triggered_at: datetime | None = None
    trigger_count: int = 0
    cooldown_minutes: int = 60          # Anti-spam cooldown

    # The condition is injected at construction (Strategy)
    _condition: AlertCondition | None = None

    model_config = {"arbitrary_types_allowed": True}

    def set_condition(self, condition: AlertCondition) -> None:
        self._condition = condition

    def evaluate(
        self,
        value: float,
        context: dict[str, Any] | None = None,
    ) -> AlertResult:
        if self._condition is None:
            raise ValueError(f"Alert '{self.name}' has no condition set.")

        triggered = (
            self.status == AlertStatus.ACTIVE
            and self._condition.evaluate(value, context)
            and self._is_past_cooldown()
        )

        result = AlertResult(
            alert_id=self.id,
            alert_name=self.name,
            triggered=triggered,
            value=value,
            condition_desc=self._condition.describe(),
            severity=self.severity,
            context=context or {},
        )

        if triggered:
            object.__setattr__(self, "last_triggered_at", datetime.now())
            object.__setattr__(self, "trigger_count", self.trigger_count + 1)
            object.__setattr__(self, "status", AlertStatus.TRIGGERED)

        return result

    def _is_past_cooldown(self) -> bool:
        if self.last_triggered_at is None:
            return True
        elapsed = (datetime.now() - self.last_triggered_at).total_seconds() / 60
        return elapsed >= self.cooldown_minutes

    def reset(self) -> None:
        object.__setattr__(self, "status", AlertStatus.ACTIVE)

    def disable(self) -> None:
        object.__setattr__(self, "status", AlertStatus.DISABLED)


# ──────────────────────────────────────────────
# AlertManager — Observer Pattern
# ──────────────────────────────────────────────

HandlerFn = Callable[[AlertResult], None]


class AlertManager:
    """
    Centralized alert registry + dispatcher (Observer).

    Usage:
        manager = AlertManager()

        # 1. Create and register an alert
        alert = Alert(name="2s10s inversion", series_id="2s10s_spread")
        alert.set_condition(ThresholdCondition(ComparisonOp.LT, threshold=-10))
        manager.register(alert)

        # 2. Subscribe handlers
        manager.subscribe(AlertSeverity.CRITICAL, send_slack_notification)
        manager.subscribe(AlertSeverity.WARNING, log_to_file)

        # 3. Evaluate (called by the scheduler)
        manager.evaluate("2s10s_spread", value=-15.2)
    """

    def __init__(self) -> None:
        self._alerts: dict[str, Alert] = {}
        self._handlers: dict[AlertSeverity | str, list[HandlerFn]] = {}
        self._history: list[AlertResult] = []

    # ── Alert management ──

    def register(self, alert: Alert) -> None:
        if alert._condition is None:
            raise ValueError(f"Alert '{alert.name}' must have a condition before registration.")
        self._alerts[alert.id] = alert

    def unregister(self, alert_id: str) -> None:
        self._alerts.pop(alert_id, None)

    def get_by_series(self, series_id: str) -> list[Alert]:
        return [a for a in self._alerts.values() if a.series_id == series_id]

    def active_alerts(self) -> list[Alert]:
        return [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]

    # ── Observers / Handlers ──

    def subscribe(
        self,
        severity: AlertSeverity | str,
        handler: HandlerFn,
    ) -> None:
        """Registers a callback for a given severity (or "all")."""
        key = severity if isinstance(severity, str) else severity
        self._handlers.setdefault(key, []).append(handler)

    def _dispatch(self, result: AlertResult) -> None:
        """Dispatches the result to subscribed handlers."""
        for key in [result.severity, "all"]:
            for handler in self._handlers.get(key, []):
                try:
                    handler(result)
                except Exception as e:
                    print(f"[AlertManager] Handler error: {e}")

    # ── Evaluation ──

    def evaluate(
        self,
        series_id: str,
        value: float,
        context: dict[str, Any] | None = None,
    ) -> list[AlertResult]:
        """Evaluates all alerts for a given series."""
        results = []
        for alert in self.get_by_series(series_id):
            result = alert.evaluate(value, context)
            self._history.append(result)
            if result.triggered:
                self._dispatch(result)
            results.append(result)
        return results

    def evaluate_all(self, data: dict[str, float]) -> list[AlertResult]:
        """Evaluates all series at once. data = {series_id: value}."""
        results = []
        for series_id, value in data.items():
            results.extend(self.evaluate(series_id, value))
        return results

    # ── History ──

    def triggered_history(self, n: int = 20) -> list[AlertResult]:
        return [r for r in self._history if r.triggered][-n:]

    def summary(self) -> str:
        lines = [
            f"── AlertManager ──",
            f"Total alerts : {len(self._alerts)}",
            f"Active       : {len(self.active_alerts())}",
            f"Total fired  : {len(self.triggered_history(1000))}",
            "",
            "Registered alerts:",
        ]
        for alert in self._alerts.values():
            cond = alert._condition.describe() if alert._condition else "no condition"
            lines.append(
                f"  [{alert.status.value:9}] {alert.name} | "
                f"{alert.series_id} | {cond} | fired {alert.trigger_count}x"
            )
        return "\n".join(lines)
