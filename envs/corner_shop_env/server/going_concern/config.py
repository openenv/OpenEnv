"""Firm configuration. A firm's starting state and parameters load from plain JSON,
which keeps the path open to real-company starts and digital twins.

All money is integer cents. All quantities are integer units. Time is in weeks."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class SKUConfig:
    id: str
    name: str
    ref_price: int            # the "fair" shelf price customers anchor on (cents)
    start_price: int
    base_demand: float        # units/week at ref price, neutral season, goodwill 1
    elasticity: float         # demand ~ (price/ref)^-elasticity
    season_amp: float         # 0..1
    season_phase: int         # week offset
    shrink_rate: float        # fraction of on-hand lost per week (spoilage/theft)
    holding_cost: int         # cents per unit on hand per week
    liquidation_frac: float   # fraction of unit cost a liquidator pays
    start_on_hand: int
    start_unit_cost: int


@dataclass
class SupplierConfig:
    id: str
    name: str
    sku: str
    unit_cost: int
    lead_time: int
    late_delivery_p: float    # chance a PO slips one extra week
    min_order: int
    terms_weeks: int          # invoice due this many weeks after delivery
    prepay: bool              # always requires payment at order time
    adversarial: bool
    short_ship_p: float       # chance a delivery arrives short
    overbill_p: float         # chance a short delivery is invoiced in full
    late_fee_rate: float      # weekly fee on overdue balance
    hold_after_weeks: int     # weeks overdue before the supplier demands prepayment
    credit_limit: int | None = None  # max owed + on order on terms; beyond it, prepay


@dataclass
class CreditConfig:
    limit: int
    apr: float
    penalty_apr: float        # added while in covenant breach
    covenant_min_current_ratio: float
    test_every_weeks: int
    call_grace_weeks: int     # weeks to repay after the line is called
    start_drawn: int


@dataclass
class StaffConfig:
    start_headcount: int
    wage: int                 # per head per week
    capacity_per_head: int    # units handled per head per week
    hire_cost: int            # recruiting cost per hire
    severance_weeks: float
    max_headcount: int


@dataclass
class EventConfig:
    demand_shock_p: float = 0.0
    supplier_disruption_p: float = 0.0
    cost_increase_p: float = 0.0
    promo_p: float = 0.0
    bulk_offer_p: float = 0.0
    scam_p: float = 0.0
    scam_legit_p: float = 0.3


@dataclass
class StartInvoice:
    supplier: str
    amount: int
    due_in_weeks: int


@dataclass
class FirmConfig:
    seed: int
    tier: int
    version: str
    name: str
    weeks: int
    start_week_of_year: int
    start_cash: int
    rent: int
    skus: list[SKUConfig]
    suppliers: list[SupplierConfig]
    credit: CreditConfig
    staff: StaffConfig
    events: EventConfig
    start_invoices: list[StartInvoice] = field(default_factory=list)
    history_weeks: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "FirmConfig":
        d = dict(d)
        d["skus"] = [SKUConfig(**s) for s in d["skus"]]
        d["suppliers"] = [SupplierConfig(**s) for s in d["suppliers"]]
        d["credit"] = CreditConfig(**d["credit"])
        d["staff"] = StaffConfig(**d["staff"])
        d["events"] = EventConfig(**d["events"])
        d["start_invoices"] = [StartInvoice(**i) for i in d.get("start_invoices", [])]
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "FirmConfig":
        return cls.from_dict(json.loads(s))

    def sku(self, sku_id: str) -> SKUConfig:
        for s in self.skus:
            if s.id == sku_id:
                return s
        raise KeyError(sku_id)

    def supplier(self, sid: str) -> SupplierConfig:
        for s in self.suppliers:
            if s.id == sid:
                return s
        raise KeyError(sid)
