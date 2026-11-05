"""
Typed models for the Cloud SRE OpenEnv environment.

Follows OpenEnv conventions: Action, Observation, State as dataclasses
inheriting from openenv.core base classes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from openenv.core.env_server import Action, Observation, State


# ─── Enums ────────────────────────────────────────────────────────────────────


class ResourceType(str, Enum):
    """Types of cloud resources available in the simulation."""
    EC2 = "ec2_instance"
    RDS = "rds_database"
    EBS = "ebs_volume"
    ALB = "alb_load_balancer"


class ResourceStatus(str, Enum):
    """Possible statuses for a cloud resource."""
    RUNNING = "running"
    STOPPED = "stopped"
    AVAILABLE = "available"       # EBS: unattached
    IN_USE = "in-use"            # EBS: attached
    REBOOTING = "rebooting"
    TERMINATED = "terminated"


class AlertSeverity(str, Enum):
    """Severity levels for monitoring alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ActionCommand(str, Enum):
    """Discrete commands the agent can issue."""
    TERMINATE = "terminate"
    SCALE = "scale"
    REBOOT = "reboot"
    INSPECT = "inspect"
    WAIT = "wait"


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class ResourceInfo:
    """Represents a single cloud resource (EC2, RDS, EBS, ALB)."""
    id: str
    name: str = ""
    type: str = ""            # ResourceType value
    status: str = ""          # ResourceStatus value
    instance_size: str = ""   # e.g., "t3.micro", "db.t3.medium"
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    cost_per_hour: float = 0.0
    attached_to: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertInfo:
    """Represents an active monitoring alert."""
    alert_id: str
    severity: str = "info"
    message: str = ""
    resource_id: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None


# ─── OpenEnv Action / Observation / State ────────────────────────────────────


from pydantic import Field

class SREAction(Action):
    """
    An action the agent submits to step().

    Attributes:
        command: One of 'terminate', 'scale', 'reboot', 'inspect', 'wait'.
        resource_id: The target resource ID (optional for 'wait').
        params: Additional parameters, e.g. {"target_size": "db.t3.medium"}.
    """
    command: str = "wait"
    resource_id: Optional[str] = None
    params: Dict[str, str] = Field(default_factory=dict)


class SREObservation(Observation):
    """
    The full observation returned by state() and step().
    Contains everything the agent can see about the infrastructure.
    """
    resources: List[dict] = Field(default_factory=list)
    alerts: List[dict] = Field(default_factory=list)
    total_hourly_cost: float = 0.0
    system_uptime: float = 100.0
    step_number: int = 0
    max_steps: int = 15
    budget_limit: Optional[float] = None
    task_description: str = ""


class SREState(State):
    """
    Episode state tracking for the Cloud SRE environment.
    Extends the base State with SRE-specific metadata.
    """
    task_id: str = ""
    current_step: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    action_count: int = 0
