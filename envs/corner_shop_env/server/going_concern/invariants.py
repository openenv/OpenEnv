"""Checks run after every step. Any failure raises InvariantViolation, which stops the run.

Three independent records must agree: the operational state (cash, layers, invoices, POs),
the double-entry ledger, and the cumulative money-flow log."""
from __future__ import annotations

from .ledger import InvariantViolation


def _fail(sim, msg: str):
    raise InvariantViolation(f"[seed={sim.cfg.seed} tier={sim.cfg.tier} week={sim.week}] {msg}")


def check(sim) -> None:
    L = sim.ledger.bal
    # money conservation: every cent has a source and destination
    if sim.cash != sim.start_cash + sim.cash_in_total - sim.cash_out_total:
        _fail(sim, f"cash {sim.cash} != start + in - out")
    if L["cash"] != sim.cash:
        _fail(sim, f"ledger cash {L['cash']} != state cash {sim.cash}")
    if sim.cash < 0:
        _fail(sim, f"negative cash {sim.cash}")
    # balance sheet
    if not sim.ledger.balances():
        _fail(sim, "balance sheet does not balance")
    if L["inventory"] != sim.inventory_value():
        _fail(sim, f"ledger inventory {L['inventory']} != layers {sim.inventory_value()}")
    if L["prepaid"] != sim.prepaid_total():
        _fail(sim, f"ledger prepaid {L['prepaid']} != open POs {sim.prepaid_total()}")
    if L["ap"] != sim.ap_total():
        _fail(sim, f"ledger AP {L['ap']} != invoices {sim.ap_total()}")
    if L["debt"] != sim.credit.drawn:
        _fail(sim, f"ledger debt {L['debt']} != drawn {sim.credit.drawn}")
    if sim.credit.drawn < 0 or sim.credit.drawn > sim.cfg.credit.limit:
        _fail(sim, f"debt {sim.credit.drawn} outside [0, {sim.cfg.credit.limit}]")
    if any(i.amount < 0 for i in sim.invoices):
        _fail(sim, "negative invoice")
    # goods conservation, no negative inventory
    for k, st in sim.skus.items():
        if any(l.qty <= 0 or l.unit_cost < 0 for l in st.layers):
            _fail(sim, f"bad inventory layer in {k}")
        expect = st.start_qty + st.received - st.sold_retail - st.sold_bulk - st.liquidated - st.shrunk
        if st.on_hand != expect:
            _fail(sim, f"goods not conserved for {k}: on hand {st.on_hand} != {expect}")
    for po in sim.pos:
        if po.qty <= 0 or po.received + po.cancelled + (po.qty if po.status == "open" else 0) != po.qty:
            _fail(sim, f"PO {po.id} quantities do not add up")
        if po.status != "open" and po.prepaid != 0:
            _fail(sim, f"closed PO {po.id} still holds prepaid")
    if sim.headcount < 0 or sim.pending_hires < 0:
        _fail(sim, "negative headcount")
    if not (0.0 < sim.goodwill <= 2.0):
        _fail(sim, f"goodwill out of range {sim.goodwill}")
