"""
Cloud SRE Environment — Server-Side Implementation.

Implements the core Environment with reset(), step(), and state() methods
following the OpenEnv specification.

Supports:
- 3 difficulty-tiered Cloud SRE tasks (easy, medium, hard)
- Seeded procedural generation for RL training
- Chaos event injection for robustness testing
- Deterministic grading with fine-grained scoring breakdowns
"""

import copy
import random
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import asdict

from openenv.core.env_server import Environment
from ..models import (
    SREAction, SREObservation, SREState,
    ResourceType, ResourceStatus, AlertSeverity, ActionCommand,
    ResourceInfo, AlertInfo,
)


# ── Helper: Name generators ────────────────────────────────────────────────

_ADJECTIVES = ["old", "legacy", "temp", "stale", "orphan", "unused", "leftover", "backup", "scratch", "test"]
_EC2_ROLES = ["web", "api", "worker", "cache", "monitor", "proxy", "gateway", "scheduler", "indexer", "renderer"]
_EBS_NOTES = ["migration-2024", "snapshot-leftover", "backup-failed", "scratch-disk", "dev-test", "canary-old"]
_ENVS_DECOY = ["deprecated", "test", "dev", "staging", "sandbox"]
_SIZES_EC2 = ["t3.nano", "t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge"]
_SIZES_LARGE = ["c5.xlarge", "c5.2xlarge", "c5.4xlarge", "m5.xlarge", "m5.2xlarge"]

RDS_PRICING = {
    "db.t3.micro": 0.017, "db.t3.small": 0.034,
    "db.t3.medium": 0.068, "db.t3.large": 0.136,
    "db.t3.xlarge": 0.272,
}

CHAOS_EVENTS = [
    {
        "type": "new_alert",
        "alert": {
            "alert_id": "chaos-cost-spike",
            "severity": "warning",
            "message": "Unexpected S3 egress cost spike detected: +$0.45/hr from cross-region transfers.",
            "metric_name": "S3EgressCost",
            "metric_value": 0.45,
        },
    },
    {
        "type": "cpu_spike",
        "description": "A random running instance's CPU spikes to 92%",
    },
    {
        "type": "new_alert",
        "alert": {
            "alert_id": "chaos-disk-warn",
            "severity": "warning",
            "message": "Disk utilization on a data volume has reached 85%. Consider expanding storage.",
            "metric_name": "DiskUtilization",
            "metric_value": 85.0,
        },
    },
    {
        "type": "cost_drift",
        "description": "Total cost drifts up slightly due to network egress charges",
    },
]


def _rand_id(rng: random.Random, prefix: str, width: int = 3) -> str:
    return f"{prefix}-{rng.randint(10**(width-1), 10**width - 1)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class Task1PhantomVolumeCleanup:
    """
    EASY: Identify and terminate unattached EBS volumes wasting money.
    Agent must NOT touch running instances or in-use volumes.
    """
    TASK_ID = "phantom_volume_cleanup"
    DIFFICULTY = "easy"
    DESCRIPTION = (
        "Your cloud account has unattached EBS volumes that are not connected "
        "to any instance but still incur charges. Identify and terminate them "
        "to reduce costs. Do NOT touch any running instances or in-use volumes."
    )

    def __init__(self):
        self.orphan_ids: Set[str] = set()

    def get_initial_state(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            self.orphan_ids = {"ebs-orphan-001", "ebs-orphan-002", "ebs-orphan-003"}
            return self._get_fixed_state()

        rng = random.Random(seed)
        num_orphans = rng.randint(2, 5)
        num_ec2 = rng.randint(3, 6)
        num_inuse_ebs = rng.randint(1, 3)

        resources = []

        # Active EC2 instances (decoys)
        ec2_ids = []
        for i in range(num_ec2):
            role = rng.choice(_EC2_ROLES)
            size = rng.choice(_SIZES_EC2)
            eid = _rand_id(rng, f"ec2-{role}")
            ec2_ids.append(eid)
            resources.append(asdict(ResourceInfo(
                id=eid, name=f"{role}-server-{i+1}",
                type=ResourceType.EC2.value, status=ResourceStatus.RUNNING.value,
                instance_size=size,
                cpu_utilization=round(rng.uniform(10, 75), 1),
                memory_utilization=round(rng.uniform(20, 80), 1),
                cost_per_hour=round(rng.uniform(0.02, 0.12), 4),
                tags={"env": "prod", "role": role},
            )))

        # In-use EBS volumes (decoys)
        for i in range(num_inuse_ebs):
            attached = rng.choice(ec2_ids) if ec2_ids else "ec2-unknown"
            resources.append(asdict(ResourceInfo(
                id=_rand_id(rng, "ebs-data"), name=f"data-vol-{i+1}",
                type=ResourceType.EBS.value, status=ResourceStatus.IN_USE.value,
                cost_per_hour=round(rng.uniform(0.05, 0.15), 4),
                attached_to=attached, tags={"env": "prod"},
            )))

        # ORPHAN EBS volumes (targets)
        for i in range(num_orphans):
            oid = _rand_id(rng, "ebs-orphan")
            self.orphan_ids.add(oid)
            resources.append(asdict(ResourceInfo(
                id=oid, name=f"{rng.choice(_ADJECTIVES)}-vol-{i+1}",
                type=ResourceType.EBS.value, status=ResourceStatus.AVAILABLE.value,
                cost_per_hour=round(rng.uniform(0.50, 2.50), 2),
                attached_to=None,
                tags={"env": rng.choice(_ENVS_DECOY), "note": rng.choice(_EBS_NOTES)},
            )))

        rng.shuffle(resources)
        total_waste = sum(r["cost_per_hour"] for r in resources
                         if r["type"] == ResourceType.EBS.value and r["status"] == ResourceStatus.AVAILABLE.value)
        total_cost = sum(r["cost_per_hour"] for r in resources)

        alerts = [asdict(AlertInfo(
            alert_id="alert-cost-001", severity=AlertSeverity.WARNING.value,
            message=f"Monthly cost projection exceeds budget. "
                    f"{num_orphans} unattached EBS volumes detected (${total_waste:.2f}/hr waste).",
            metric_name="CostAnomaly", metric_value=round(total_waste, 2),
        ))]

        return {
            "resources": resources, "alerts": alerts,
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 100.0, "budget_limit": None,
        }

    def _get_fixed_state(self) -> Dict[str, Any]:
        resources = [
            asdict(ResourceInfo(id="ec2-web-001", name="web-server-1", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="t3.medium",
                                cpu_utilization=45.0, memory_utilization=62.0,
                                cost_per_hour=0.0416, tags={"env": "prod", "role": "web"})),
            asdict(ResourceInfo(id="ec2-web-002", name="web-server-2", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="t3.medium",
                                cpu_utilization=38.0, memory_utilization=55.0,
                                cost_per_hour=0.0416, tags={"env": "prod", "role": "web"})),
            asdict(ResourceInfo(id="ec2-api-001", name="api-server-1", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="t3.large",
                                cpu_utilization=60.0, memory_utilization=70.0,
                                cost_per_hour=0.0832, tags={"env": "prod", "role": "api"})),
            asdict(ResourceInfo(id="ebs-data-001", name="web-data-vol", type=ResourceType.EBS.value,
                                status=ResourceStatus.IN_USE.value, cost_per_hour=0.10,
                                attached_to="ec2-web-001", tags={"env": "prod"})),
            asdict(ResourceInfo(id="ebs-orphan-001", name="old-migration-vol", type=ResourceType.EBS.value,
                                status=ResourceStatus.AVAILABLE.value, cost_per_hour=1.40,
                                tags={"env": "deprecated", "note": "migration-2024"})),
            asdict(ResourceInfo(id="ebs-orphan-002", name="test-snapshot-vol", type=ResourceType.EBS.value,
                                status=ResourceStatus.AVAILABLE.value, cost_per_hour=1.40,
                                tags={"env": "test", "note": "snapshot-leftover"})),
            asdict(ResourceInfo(id="ebs-orphan-003", name="dev-scratch-vol", type=ResourceType.EBS.value,
                                status=ResourceStatus.AVAILABLE.value, cost_per_hour=1.40,
                                tags={"env": "dev", "note": "scratch-disk"})),
        ]
        alerts = [asdict(AlertInfo(
            alert_id="alert-cost-001", severity=AlertSeverity.WARNING.value,
            message="Monthly cost projection exceeds budget by 15%. "
                    "3 unattached EBS volumes detected ($4.20/hr waste).",
            metric_name="CostAnomaly", metric_value=4.20,
        ))]
        total_cost = sum(r["cost_per_hour"] for r in resources)
        return {
            "resources": resources, "alerts": alerts,
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 100.0, "budget_limit": None,
        }

    def grade(self, action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        score = 0.0
        breakdown = {
            "orphans_terminated": [], "active_resources_terminated": [],
            "total_orphans": len(self.orphan_ids), "penalties": 0.0,
        }
        terminated_ids = {e["resource_id"] for e in action_history
                         if e.get("command") == "terminate" and e.get("resource_id")}
        per_orphan = 1.0 / len(self.orphan_ids) if self.orphan_ids else 0.33
        for rid in terminated_ids:
            if rid in self.orphan_ids:
                score += per_orphan
                breakdown["orphans_terminated"].append(rid)
            else:
                score -= 0.50
                breakdown["active_resources_terminated"].append(rid)
                breakdown["penalties"] += 0.50

        score = round(max(0.0, min(1.0, score)), 2)
        if (len(breakdown["orphans_terminated"]) == len(self.orphan_ids)
                and not breakdown["active_resources_terminated"]):
            score = 1.0
        return score, breakdown


class Task2LatencySpikeRemediation:
    """
    MEDIUM: Scale an under-provisioned RDS to fix high API latency, within budget.
    """
    TASK_ID = "latency_spike_remediation"
    DIFFICULTY = "medium"
    DESCRIPTION = (
        "A critical latency alert has fired. Your RDS database is running on a "
        "tiny instance and cannot handle the current load. Scale it up to fix "
        "latency, but keep total hourly cost under the budget limit."
    )
    BUDGET_LIMIT = 12.00

    def __init__(self):
        self.rds_id: str = "rds-primary-001"

    def get_initial_state(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            self.rds_id = "rds-primary-001"
            return self._get_fixed_state()

        rng = random.Random(seed)
        num_ec2 = rng.randint(3, 6)
        resources = []

        for i in range(num_ec2):
            role = rng.choice(["app", "cache", "monitor", "proxy", "gateway"])
            resources.append(asdict(ResourceInfo(
                id=_rand_id(rng, f"ec2-{role}"), name=f"{role}-server-{i+1}",
                type=ResourceType.EC2.value, status=ResourceStatus.RUNNING.value,
                instance_size=rng.choice(_SIZES_EC2[2:]),
                cpu_utilization=round(rng.uniform(25, 75), 1),
                memory_utilization=round(rng.uniform(30, 80), 1),
                cost_per_hour=round(rng.uniform(0.04, 0.15), 4),
                tags={"env": "prod", "role": role},
            )))

        rds_id = _rand_id(rng, "rds-primary")
        self.rds_id = rds_id
        rds_cpu = round(rng.uniform(92, 99), 1)
        resources.append(asdict(ResourceInfo(
            id=rds_id, name="primary-db", type=ResourceType.RDS.value,
            status=ResourceStatus.RUNNING.value, instance_size="db.t3.micro",
            cpu_utilization=rds_cpu, memory_utilization=round(rng.uniform(88, 98), 1),
            cost_per_hour=0.017, tags={"env": "prod", "role": "database"},
        )))

        resources.append(asdict(ResourceInfo(
            id=_rand_id(rng, "alb-main"), name="main-load-balancer",
            type=ResourceType.ALB.value, status=ResourceStatus.RUNNING.value,
            cost_per_hour=0.0225, tags={"env": "prod", "role": "lb"},
        )))

        rng.shuffle(resources)
        latency_val = round(rng.uniform(1800, 3000), 0)
        alerts = [
            asdict(AlertInfo(alert_id="alert-latency-001", severity=AlertSeverity.CRITICAL.value,
                             message=f"API p99 latency has exceeded {latency_val:.0f}ms. "
                                     f"Root cause: RDS '{rds_id}' (db.t3.micro) at {rds_cpu}% CPU.",
                             resource_id=rds_id, metric_name="P99Latency", metric_value=latency_val)),
            asdict(AlertInfo(alert_id="alert-cpu-001", severity=AlertSeverity.WARNING.value,
                             message=f"RDS '{rds_id}' CPU utilization is at {rds_cpu}%.",
                             resource_id=rds_id, metric_name="CPUUtilization", metric_value=rds_cpu)),
        ]
        total_cost = sum(r["cost_per_hour"] for r in resources)
        budget = round(total_cost + rng.uniform(3.0, 8.0), 2)
        return {
            "resources": resources, "alerts": alerts,
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": round(rng.uniform(65, 82), 1),
            "budget_limit": budget,
        }

    def _get_fixed_state(self) -> Dict[str, Any]:
        resources = [
            asdict(ResourceInfo(id="ec2-app-001", name="app-server-1", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="t3.large",
                                cpu_utilization=72.0, memory_utilization=65.0,
                                cost_per_hour=0.0832, tags={"env": "prod", "role": "app"})),
            asdict(ResourceInfo(id="ec2-app-002", name="app-server-2", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="t3.large",
                                cpu_utilization=68.0, memory_utilization=60.0,
                                cost_per_hour=0.0832, tags={"env": "prod", "role": "app"})),
            asdict(ResourceInfo(id="rds-primary-001", name="primary-db", type=ResourceType.RDS.value,
                                status=ResourceStatus.RUNNING.value, instance_size="db.t3.micro",
                                cpu_utilization=98.0, memory_utilization=95.0,
                                cost_per_hour=0.017, tags={"env": "prod", "role": "database"})),
            asdict(ResourceInfo(id="alb-main-001", name="main-load-balancer", type=ResourceType.ALB.value,
                                status=ResourceStatus.RUNNING.value,
                                cost_per_hour=0.0225, tags={"env": "prod", "role": "lb"})),
        ]
        alerts = [
            asdict(AlertInfo(alert_id="alert-latency-001", severity=AlertSeverity.CRITICAL.value,
                             message="API p99 latency has exceeded 2000ms. Root cause: RDS 'primary-db' at 98% CPU.",
                             resource_id="rds-primary-001", metric_name="P99Latency", metric_value=2150.0)),
        ]
        total_cost = sum(r["cost_per_hour"] for r in resources)
        return {
            "resources": resources, "alerts": alerts,
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 78.0, "budget_limit": self.BUDGET_LIMIT,
        }

    def grade(self, action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        score = 0.0
        breakdown = {"rds_scaled": False, "under_budget": False, "alert_resolved": False,
                     "ec2s_terminated": [], "prod_terminated": [], "penalties": 0.0}
        prod_ids = {r["id"] for r in initial_state["resources"]
                    if r.get("tags", {}).get("env") == "prod"}
        valid_sizes = ["db.t3.medium", "db.t3.large", "db.t3.xlarge"]

        for entry in action_history:
            cmd, rid = entry.get("command"), entry.get("resource_id")
            params = entry.get("params", {})
            if cmd == "scale" and rid == self.rds_id and params.get("target_size", "") in valid_sizes:
                breakdown["rds_scaled"] = True
            if cmd == "terminate" and rid and rid.startswith("ec2"):
                breakdown["ec2s_terminated"].append(rid)
            if cmd == "terminate" and rid in prod_ids:
                breakdown["prod_terminated"].append(rid)

        if breakdown["rds_scaled"]:
            score += 0.40
            breakdown["alert_resolved"] = True
            score += 0.30
        budget = initial_state.get("budget_limit", self.BUDGET_LIMIT)
        if final_state.get("total_hourly_cost", 999) <= budget:
            breakdown["under_budget"] = True
            score += 0.30
        for _ in breakdown["prod_terminated"]:
            score -= 0.30
            breakdown["penalties"] += 0.30

        return round(max(0.0, min(1.0, score)), 2), breakdown


class Task3NoisyNeighborIncident:
    """
    HARD: A rogue test EC2 instance has crashed a prod backend.
    Agent must investigate, terminate the rogue, and reboot production.
    """
    TASK_ID = "noisy_neighbor_incident"
    DIFFICULTY = "hard"
    DESCRIPTION = (
        "CRITICAL INCIDENT: A rogue EC2 instance (tagged env:test) is consuming "
        "excessive resources and has caused the production backend to crash. "
        "Investigate, terminate the rogue, and restore the production backend."
    )

    def __init__(self):
        self.rogue_id: str = "ec2-rogue-test-001"
        self.backend_id: str = "ec2-backend-prod-001"
        self.prod_ids: Set[str] = set()

    def get_initial_state(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            self.rogue_id = "ec2-rogue-test-001"
            self.backend_id = "ec2-backend-prod-001"
            self.prod_ids = {"ec2-frontend-001", "ec2-frontend-002",
                             "ec2-api-prod-001", "ec2-backend-prod-001", "rds-prod-001"}
            return self._get_fixed_state()

        rng = random.Random(seed)
        resources = []

        # THE ROGUE INSTANCE
        rogue_id = _rand_id(rng, "ec2-rogue-test")
        self.rogue_id = rogue_id
        resources.append(asdict(ResourceInfo(
            id=rogue_id, name="load-test-runner", type=ResourceType.EC2.value,
            status=ResourceStatus.RUNNING.value, instance_size=rng.choice(_SIZES_LARGE),
            cpu_utilization=100.0, memory_utilization=round(rng.uniform(85, 98), 1),
            cost_per_hour=round(rng.uniform(0.40, 1.20), 2),
            tags={"env": "test", "role": "load-testing", "owner": "qa-team", "note": "forgot to stop"},
        )))

        # CRASHED PROD BACKEND
        backend_id = _rand_id(rng, "ec2-backend-prod")
        self.backend_id = backend_id
        self.prod_ids = {backend_id}
        resources.append(asdict(ResourceInfo(
            id=backend_id, name="backend-primary", type=ResourceType.EC2.value,
            status=ResourceStatus.STOPPED.value, instance_size="m5.xlarge",
            cost_per_hour=0.192,
            tags={"env": "prod", "role": "backend", "critical": "true"},
        )))

        # Normal prod instances
        for i in range(rng.randint(3, 6)):
            role = rng.choice(["frontend", "api", "db-proxy", "gateway", "worker"])
            pid = _rand_id(rng, f"ec2-{role}-prod")
            self.prod_ids.add(pid)
            resources.append(asdict(ResourceInfo(
                id=pid, name=f"{role}-{i+1}", type=ResourceType.EC2.value,
                status=ResourceStatus.RUNNING.value,
                instance_size=rng.choice(_SIZES_EC2[2:]),
                cpu_utilization=round(rng.uniform(20, 75), 1),
                memory_utilization=round(rng.uniform(25, 70), 1),
                cost_per_hour=round(rng.uniform(0.03, 0.10), 4),
                tags={"env": "prod", "role": role},
            )))

        rng.shuffle(resources)
        rogue_cost = next(r["cost_per_hour"] for r in resources if r["id"] == rogue_id)
        alerts = [
            asdict(AlertInfo(alert_id="alert-crit-001", severity=AlertSeverity.CRITICAL.value,
                             message=f"Production backend '{backend_id}' is DOWN. HTTP 503 errors spiking.",
                             resource_id=backend_id, metric_name="HealthCheck", metric_value=0.0)),
            asdict(AlertInfo(alert_id="alert-crit-002", severity=AlertSeverity.CRITICAL.value,
                             message=f"Abnormal CPU usage: '{rogue_id}' consuming 100% CPU. Investigate immediately.",
                             resource_id=rogue_id, metric_name="CPUUtilization", metric_value=100.0)),
            asdict(AlertInfo(alert_id="alert-cost-002", severity=AlertSeverity.WARNING.value,
                             message=f"Hourly cost spike: ${rogue_cost:.2f}/hr from a single test instance.",
                             resource_id=rogue_id, metric_name="CostAnomaly", metric_value=rogue_cost)),
        ]
        total_cost = sum(r["cost_per_hour"] for r in resources)
        return {
            "resources": resources, "alerts": alerts,
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": round(rng.uniform(25, 45), 1), "budget_limit": None,
        }

    def _get_fixed_state(self) -> Dict[str, Any]:
        resources = [
            asdict(ResourceInfo(id="ec2-rogue-test-001", name="load-test-runner", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="c5.4xlarge",
                                cpu_utilization=100.0, memory_utilization=92.0, cost_per_hour=0.68,
                                tags={"env": "test", "role": "load-testing", "owner": "qa-team"})),
            asdict(ResourceInfo(id="ec2-backend-prod-001", name="backend-primary", type=ResourceType.EC2.value,
                                status=ResourceStatus.STOPPED.value, instance_size="m5.xlarge",
                                cost_per_hour=0.192,
                                tags={"env": "prod", "role": "backend", "critical": "true"})),
            asdict(ResourceInfo(id="ec2-frontend-001", name="frontend-1", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="t3.medium",
                                cpu_utilization=55.0, memory_utilization=40.0,
                                cost_per_hour=0.0416, tags={"env": "prod", "role": "frontend"})),
            asdict(ResourceInfo(id="ec2-api-prod-001", name="api-gateway", type=ResourceType.EC2.value,
                                status=ResourceStatus.RUNNING.value, instance_size="t3.large",
                                cpu_utilization=75.0, memory_utilization=60.0,
                                cost_per_hour=0.0832, tags={"env": "prod", "role": "api"})),
            asdict(ResourceInfo(id="rds-prod-001", name="prod-database", type=ResourceType.RDS.value,
                                status=ResourceStatus.RUNNING.value, instance_size="db.r5.large",
                                cpu_utilization=40.0, memory_utilization=55.0,
                                cost_per_hour=0.24, tags={"env": "prod", "role": "database"})),
        ]
        alerts = [
            asdict(AlertInfo(alert_id="alert-crit-001", severity=AlertSeverity.CRITICAL.value,
                             message="Production backend 'backend-primary' is DOWN. HTTP 503 errors spiking.",
                             resource_id="ec2-backend-prod-001", metric_name="HealthCheck", metric_value=0.0)),
            asdict(AlertInfo(alert_id="alert-crit-002", severity=AlertSeverity.CRITICAL.value,
                             message="Abnormal CPU usage: 'ec2-rogue-test-001' consuming 100% CPU.",
                             resource_id="ec2-rogue-test-001", metric_name="CPUUtilization", metric_value=100.0)),
        ]
        total_cost = sum(r["cost_per_hour"] for r in resources)
        return {
            "resources": resources, "alerts": alerts,
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 35.0, "budget_limit": None,
        }

    def grade(self, action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        score = 0.0
        breakdown = {"inspected_rogue": False, "terminated_rogue": False,
                     "rebooted_backend": False, "alerts_resolved": False,
                     "prod_terminated": [], "penalties": 0.0}

        for entry in action_history:
            cmd, rid = entry.get("command"), entry.get("resource_id")
            if cmd == "inspect" and rid == self.rogue_id and not breakdown["terminated_rogue"]:
                breakdown["inspected_rogue"] = True
            if cmd == "terminate" and rid == self.rogue_id:
                breakdown["terminated_rogue"] = True
            if cmd == "reboot" and rid == self.backend_id:
                breakdown["rebooted_backend"] = True
            if cmd == "terminate" and rid in self.prod_ids:
                breakdown["prod_terminated"].append(rid)

        if breakdown["inspected_rogue"]:
            score += 0.20
        if breakdown["terminated_rogue"]:
            score += 0.30
        if breakdown["rebooted_backend"]:
            score += 0.30
        if breakdown["terminated_rogue"] and breakdown["rebooted_backend"]:
            breakdown["alerts_resolved"] = True
            score += 0.20
        for _ in breakdown["prod_terminated"]:
            score -= 0.50
            breakdown["penalties"] += 0.50

        return round(max(0.0, min(1.0, score)), 2), breakdown


TASK_REGISTRY = {
    "phantom_volume_cleanup": Task1PhantomVolumeCleanup,
    "latency_spike_remediation": Task2LatencySpikeRemediation,
    "noisy_neighbor_incident": Task3NoisyNeighborIncident,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════


class CloudSREEnvironment(Environment):
    """
    OpenEnv-compliant Cloud SRE & FinOps Environment.

    The agent manages a simulated cloud infrastructure: diagnosing outages,
    terminating idle resources, scaling services, and optimizing costs
    without causing collateral damage to production workloads.

    Supports 3 difficulty-tiered tasks with seeded procedural generation
    and chaos event injection for robustness testing.
    """

    MAX_STEPS = 15

    def __init__(self):
        super().__init__()
        self._env_state: Dict[str, Any] = {}
        self._initial_state: Dict[str, Any] = {}
        self._action_history: List[Dict[str, Any]] = []
        self._current_step: int = 0
        self._done: bool = False
        self._task_id: str = ""
        self._task = None
        self._cumulative_reward: float = 0.0
        self._seed: Optional[int] = None
        self._chaos_enabled: bool = False
        self._cost_drift: float = 0.0
        self._sre_state = SREState()

    def reset(
        self,
        task_id: str = "phantom_volume_cleanup",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> SREObservation:
        """Reset to a specific task's initial state."""
        task_cls = TASK_REGISTRY.get(task_id)
        if task_cls is None:
            raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}")

        self._task = task_cls()
        self._task_id = task_id
        self._seed = seed
        self._initial_state = self._task.get_initial_state(seed=seed)
        self._env_state = copy.deepcopy(self._initial_state)
        self._action_history = []
        self._current_step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._chaos_enabled = seed is not None
        self._cost_drift = 0.0

        self._sre_state = SREState(
            episode_id=episode_id if episode_id is not None else str(uuid.uuid4()),
            task_id=task_id,
        )

        return self._build_observation()

    def step(self, action: SREAction) -> SREObservation:
        """Execute one agent action. Returns the resulting observation."""
        if self._done:
            return self._build_observation()

        self._current_step += 1
        step_reward = 0.0

        # Dispatch action
        cmd = action.command
        rid = action.resource_id
        params = action.params or {}

        if cmd == ActionCommand.TERMINATE.value:
            step_reward, _ = self._handle_terminate(rid)
        elif cmd == ActionCommand.SCALE.value:
            step_reward, _ = self._handle_scale(rid, params.get("target_size", ""))
        elif cmd == ActionCommand.REBOOT.value:
            step_reward, _ = self._handle_reboot(rid)
        elif cmd == ActionCommand.INSPECT.value:
            step_reward, _ = self._handle_inspect(rid)
        elif cmd == ActionCommand.WAIT.value:
            step_reward = -0.01

        self._action_history.append({
            "step": self._current_step,
            "command": cmd, "resource_id": rid, "params": params,
        })

        # Chaos injection
        if self._chaos_enabled and self._seed is not None:
            self._maybe_inject_chaos()

        # Recalculate
        self._recalculate_state()

        if self._current_step >= self.MAX_STEPS:
            self._done = True

        self._cumulative_reward += step_reward
        self._sre_state.current_step = self._current_step
        self._sre_state.step_count = self._current_step
        self._sre_state.done = self._done
        self._sre_state.cumulative_reward = round(self._cumulative_reward, 4)
        self._sre_state.action_count = len(self._action_history)

        return self._build_observation(reward=step_reward)

    @property
    def state(self) -> SREState:
        """Return current episode state."""
        return self._sre_state

    def grade(self) -> Tuple[float, Dict]:
        """Run the deterministic grader for the current task."""
        if self._task is None:
            return 0.0, {"error": "No task loaded"}
        return self._task.grade(self._action_history, self._env_state, self._initial_state)

    # ── Observation builder ──

    def _build_observation(self, reward: float = 0.0) -> SREObservation:
        return SREObservation(
            reward=reward,
            done=self._done,
            resources=self._env_state.get("resources", []),
            alerts=self._env_state.get("alerts", []),
            total_hourly_cost=self._env_state.get("total_hourly_cost", 0.0),
            system_uptime=self._env_state.get("system_uptime", 100.0),
            step_number=self._current_step,
            max_steps=self.MAX_STEPS,
            budget_limit=self._env_state.get("budget_limit"),
            task_description=self._task.DESCRIPTION if self._task else "",
        )

    # ── Action handlers ──

    def _find_resource(self, resource_id: str) -> Optional[Dict]:
        for r in self._env_state.get("resources", []):
            if r["id"] == resource_id:
                return r
        return None

    def _resolve_alerts_for(self, resource_id: str):
        self._env_state["alerts"] = [
            a for a in self._env_state.get("alerts", [])
            if a.get("resource_id") != resource_id
        ]

    def _handle_terminate(self, resource_id: Optional[str]) -> Tuple[float, str]:
        if not resource_id:
            return -0.05, "Error: No resource_id"
        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: '{resource_id}' not found"

        tags = resource.get("tags", {})
        cost = resource.get("cost_per_hour", 0)
        is_prod = tags.get("env") == "prod"
        is_attached_ebs = (resource.get("type") == ResourceType.EBS.value
                           and resource.get("status") == ResourceStatus.IN_USE.value)

        if is_prod and resource.get("type") in (
            ResourceType.EC2.value, ResourceType.RDS.value, ResourceType.ALB.value
        ):
            reward = -0.15
        elif is_attached_ebs:
            reward = -0.10
        elif resource.get("status") == ResourceStatus.AVAILABLE.value:
            reward = 0.05 + (cost * 0.02)
        else:
            reward = 0.02

        self._env_state["resources"] = [r for r in self._env_state["resources"] if r["id"] != resource_id]
        self._resolve_alerts_for(resource_id)
        return reward, f"Terminated '{resource_id}'"

    def _handle_scale(self, resource_id: Optional[str], target_size: str) -> Tuple[float, str]:
        if not resource_id or not target_size:
            return -0.05, "Error: Missing resource_id or target_size"
        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: '{resource_id}' not found"

        old_size = resource.get("instance_size", "unknown")
        pre_mutation_cpu = resource.get("cpu_utilization", 0)

        if resource.get("type") == ResourceType.RDS.value:
            if target_size not in RDS_PRICING:
                return -0.05, f"Error: Invalid RDS size '{target_size}'"
            new_cost = RDS_PRICING[target_size]
        else:
            new_cost = resource.get("cost_per_hour", 0) * 2.0

        for r in self._env_state["resources"]:
            if r["id"] == resource_id:
                r["instance_size"] = target_size
                r["cost_per_hour"] = new_cost
                if r.get("cpu_utilization", 0) > 80:
                    r["cpu_utilization"] = 45.0
                break

        self._resolve_alerts_for(resource_id)
        reward = 0.08
        if pre_mutation_cpu > 80:
            reward += 0.05
        return reward, f"Scaled '{resource_id}' from {old_size} to {target_size}"

    def _handle_reboot(self, resource_id: Optional[str]) -> Tuple[float, str]:
        if not resource_id:
            return -0.05, "Error: No resource_id"
        for r in self._env_state.get("resources", []):
            if r["id"] == resource_id:
                if r["status"] == ResourceStatus.STOPPED.value:
                    r["status"] = ResourceStatus.RUNNING.value
                    r["cpu_utilization"] = 15.0
                    r["memory_utilization"] = 20.0
                    self._env_state["system_uptime"] = min(
                        100.0, self._env_state.get("system_uptime", 0) + 30.0)
                    self._resolve_alerts_for(resource_id)
                    return 0.10, f"Rebooted '{resource_id}' — now RUNNING"
                elif r["status"] == ResourceStatus.RUNNING.value:
                    r["cpu_utilization"] = 10.0
                    return -0.02, f"Rebooted '{resource_id}' — temporary disruption"
                else:
                    return -0.05, f"Cannot reboot '{resource_id}' in state '{r['status']}'"
        return -0.05, f"Error: '{resource_id}' not found"

    def _handle_inspect(self, resource_id: Optional[str]) -> Tuple[float, str]:
        if not resource_id:
            return -0.01, "Error: No resource_id"
        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.01, f"Error: '{resource_id}' not found"
        return 0.01, f"Inspected '{resource_id}'"

    def _recalculate_state(self):
        self._env_state["total_hourly_cost"] = round(
            sum(r.get("cost_per_hour", 0) for r in self._env_state.get("resources", []))
            + self._cost_drift,
            4,
        )
        critical = [a for a in self._env_state.get("alerts", []) if a.get("severity") == "critical"]
        if not critical:
            self._env_state["system_uptime"] = min(100.0, self._env_state.get("system_uptime", 100) + 10.0)
        else:
            self._env_state["system_uptime"] = max(0.0, self._env_state.get("system_uptime", 100) - 5.0)

    def _maybe_inject_chaos(self):
        rng = random.Random(self._seed * 1000 + self._current_step)
        if self._current_step < 4 or rng.random() > 0.25:
            return
        event = rng.choice(CHAOS_EVENTS)
        if event["type"] == "new_alert":
            alert = event["alert"].copy()
            alert["alert_id"] = f"{alert['alert_id']}-step{self._current_step}"
            self._env_state.setdefault("alerts", []).append(alert)
        elif event["type"] == "cpu_spike":
            running = [r for r in self._env_state.get("resources", [])
                       if r.get("status") == "running" and r.get("cpu_utilization", 0) < 80]
            if running:
                target = rng.choice(running)
                target["cpu_utilization"] = round(rng.uniform(88, 96), 1)
        elif event["type"] == "cost_drift":
            drift = round(rng.uniform(0.05, 0.20), 4)
            self._cost_drift = round(self._cost_drift + drift, 4)
