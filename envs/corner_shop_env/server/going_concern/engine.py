"""The firm simulation. One step is one week.

Order of a week:
  1. start of week: new hires join, scheduled supplier price changes apply, stale offers expire
  2. the agent's action (prices, borrow/repay, payments, disputes, fire, hire, accept offers,
     orders, liquidations, replies)
  3. deliveries due this week arrive (and are invoiced, or settle a prepayment)
  4. accepted bulk customer orders due this week ship
  5. retail demand is drawn; sales limited by stock and staff capacity
  6. shrink (spoilage/theft)
  7. mandatory costs: payroll, rent, holding, interest, a called loan
  8. overdue invoices take late fees; suppliers owed too long put the firm on credit hold
  9. covenant test (every few weeks)
 10. customer goodwill updates from fill rate and pricing
 11. random events for next week (shocks, offers, notices)

Money is integer cents; goods are integer units. Every change goes through a method that
updates the operational state, the double-entry ledger and the money-flow log together, and
`invariants.check` reconciles all three after every step.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from .config import FirmConfig, SupplierConfig
from .ledger import Ledger
from .rng import make_rng, poisson, stoch_round
from .version import ENGINE_VERSION

STREAMS = ("demand", "supply", "events", "shrink")


class Bankruptcy(Exception):
    def __init__(self, msg: str, unpaid: int = 0):
        super().__init__(msg)
        self.unpaid = unpaid  # the mandatory payment that could not be made (cents)


@dataclass
class Layer:
    qty: int
    unit_cost: int


@dataclass
class PurchaseOrder:
    id: str
    supplier: str
    sku: str
    qty: int
    unit_cost: int
    placed_week: int
    eta_week: int
    prepaid: int = 0
    status: str = "open"  # open | received | lost
    received: int = 0
    cancelled: int = 0
    scam: bool = False


@dataclass
class Invoice:
    id: str
    supplier: str
    po: str
    amount: int           # outstanding
    original: int
    issued_week: int
    due_week: int
    overbilled: int = 0
    late_fees: int = 0
    disputed: bool = False

    def overdue_weeks(self, week: int) -> int:
        if self.amount <= 0 or week < self.due_week:
            return 0
        return week - self.due_week + 1


@dataclass
class Message:
    id: str
    week: int
    sender: str
    kind: str
    subject: str
    body: str
    expires_week: int | None = None
    data: dict = field(default_factory=dict)
    status: str = "open"  # open | accepted | expired | info


@dataclass
class SkuState:
    price: int
    layers: list[Layer]
    start_qty: int
    received: int = 0
    sold_retail: int = 0
    sold_bulk: int = 0
    liquidated: int = 0
    shrunk: int = 0
    sales_history: list[int] = field(default_factory=list)
    demand_last: int = 0     # hidden from the agent
    sold_last: int = 0
    stockout_last: bool = False

    @property
    def on_hand(self) -> int:
        return sum(l.qty for l in self.layers)

    @property
    def value(self) -> int:
        return sum(l.qty * l.unit_cost for l in self.layers)


@dataclass
class SupplierState:
    unit_cost: int
    pending_costs: list[tuple[int, int]] = field(default_factory=list)  # (week, new cost)
    extra_lead: int = 0
    extra_lead_until: int = -1
    promo: dict | None = None  # {"week", "min_qty", "discount"}


@dataclass
class CreditState:
    drawn: int
    frozen: bool = False
    called_due_week: int | None = None
    in_breach: bool = False
    breach_streak: int = 0
    breaches: int = 0
    last_ratio: float | None = None


class Sim:
    def __init__(self, cfg: FirmConfig):
        self.cfg = cfg
        self.engine_version = ENGINE_VERSION
        self.week = 0
        self.horizon = cfg.weeks
        self.done = False
        self.bankrupt = False
        self.bankrupt_reason = ""
        self.bankrupt_unpaid = 0
        self._tag: tuple = ()
        self.rng = {n: make_rng(cfg.seed, cfg.tier, cfg.version, ENGINE_VERSION, n) for n in STREAMS}
        self.ledger = Ledger()
        self.goodwill = 1.0
        self._ids = {"PO": 0, "INV": 0, "M": 0}
        self.cash = 0
        self.cash_in_total = 0
        self.cash_out_total = 0
        self.flows: list[tuple] = []        # this week's (src, dst, cents, kind)
        self.events: list[dict] = []        # this week's notable events (for logs/viewer)
        self.counters = {"frivolous_disputes": 0, "auto_draws": 0, "stockout_weeks": 0,
                         "lost_units": 0, "late_fee_total": 0}
        self.messages: list[Message] = []
        self.pos: list[PurchaseOrder] = []
        self.invoices: list[Invoice] = []
        self.commitments: list[dict] = []   # accepted bulk orders
        self.demand_shocks: list[dict] = []
        self.replies: list[dict] = []
        self.headcount = cfg.staff.start_headcount
        self.pending_hires = 0
        self.sup = {s.id: SupplierState(unit_cost=s.unit_cost) for s in cfg.suppliers}
        # what customers consider a fair price; drifts with the market when supplier costs rise
        self.ref_price = {s.id: float(s.ref_price) for s in cfg.skus}
        self.credit = CreditState(drawn=0)
        self.skus: dict[str, SkuState] = {}
        self._init_balance_sheet()
        self._init_history()

    # ------------------------------------------------------------------ setup
    def _next_id(self, kind: str) -> str:
        self._ids[kind] += 1
        return f"{kind}-{self._ids[kind]}"

    def _init_balance_sheet(self) -> None:
        cfg = self.cfg
        inv_value = 0
        for s in cfg.skus:
            layers = [Layer(s.start_on_hand, s.start_unit_cost)] if s.start_on_hand > 0 else []
            self.skus[s.id] = SkuState(price=s.start_price, layers=layers, start_qty=s.start_on_hand)
            inv_value += s.start_on_hand * s.start_unit_cost
        ap = 0
        for si in cfg.start_invoices:
            inv = Invoice(id=self._next_id("INV"), supplier=si.supplier, po="opening",
                          amount=si.amount, original=si.amount, issued_week=-1,
                          due_week=si.due_in_weeks)
            self.invoices.append(inv)
            ap += si.amount
        drawn = cfg.credit.start_drawn
        self.cash = cfg.start_cash
        self.credit.drawn = drawn
        capital = cfg.start_cash + inv_value - ap - drawn
        debits = {"cash": cfg.start_cash, "inventory": inv_value}
        credits = {"ap": ap, "debt": drawn, "capital": capital}
        if capital < 0:
            debits["capital"] = -capital
            del credits["capital"]
        self.ledger.post("opening balance", debits, credits)
        self.start_cash = cfg.start_cash

    def _init_history(self) -> None:
        """Synthetic pre-episode sales history so the agent sees a demand signal on day one."""
        rng = make_rng(self.cfg.seed, self.cfg.tier, self.cfg.version, ENGINE_VERSION, "history")
        for s in self.cfg.skus:
            st = self.skus[s.id]
            for k in range(self.cfg.history_weeks, 0, -1):
                lam = self._lambda(s.id, -k, ignore_shocks=True)
                st.sales_history.append(poisson(rng, lam))
            st.sold_last = st.sales_history[-1] if st.sales_history else 0

    # ------------------------------------------------------------------ cloning (going-concern rollouts)
    def clone(self) -> "Sim":
        return copy.deepcopy(self)

    def reseed(self, *tag) -> None:
        self._tag = tuple(tag)
        self.rng = {n: make_rng(self.cfg.seed, self.cfg.tier, self.cfg.version, ENGINE_VERSION, *tag, n)
                    for n in STREAMS}

    def _cell_rng(self, stream: str, week: int, key: str):
        """A random stream per (stream, week, key): draws never depend on earlier actions
        (common random numbers across policies on the same seed)."""
        return make_rng(self.cfg.seed, self.cfg.tier, self.cfg.version, ENGINE_VERSION, *self._tag, stream, week, key)

    # ------------------------------------------------------------------ primitive operations
    def _cash_move(self, amount: int, counterparty: str, kind: str) -> None:
        """amount > 0: money into the firm; < 0: out of the firm."""
        if amount == 0:
            return
        if amount > 0:
            self.cash_in_total += amount
            self.flows.append((counterparty, "firm", amount, kind))
        else:
            self.cash_out_total += -amount
            self.flows.append(("firm", counterparty, -amount, kind))
        self.cash += amount

    def _remove_fifo(self, sku: str, qty: int) -> int:
        st = self.skus[sku]
        if qty > st.on_hand or qty < 0:
            raise ValueError(f"remove {qty} of {sku} with {st.on_hand} on hand")
        cost, need = 0, qty
        while need > 0:
            layer = st.layers[0]
            take = min(need, layer.qty)
            cost += take * layer.unit_cost
            layer.qty -= take
            need -= take
            if layer.qty == 0:
                st.layers.pop(0)
        return cost

    def _borrow(self, amount: int, auto: bool = False) -> None:
        self.credit.drawn += amount
        self._cash_move(amount, "bank", "auto_draw" if auto else "draw")
        self.ledger.post("draw", {"cash": amount}, {"debt": amount})
        if auto:
            self.counters["auto_draws"] += 1

    def _repay(self, amount: int) -> None:
        self.credit.drawn -= amount
        self._cash_move(-amount, "bank", "repay")
        self.ledger.post("repay", {"debt": amount}, {"cash": amount})

    def available_credit(self) -> int:
        if self.credit.frozen:
            return 0
        return max(0, self.cfg.credit.limit - self.credit.drawn)

    def _pay_mandatory(self, amount: int, payee: str, acct: str) -> None:
        if amount <= 0:
            return
        if self.cash < amount:
            short = amount - self.cash
            if short <= self.available_credit():
                self._borrow(short, auto=True)
            else:
                raise Bankruptcy(f"could not pay {acct} of ${amount / 100:,.2f} "
                                 f"(cash ${self.cash / 100:,.2f}, credit available "
                                 f"${self.available_credit() / 100:,.2f})", unpaid=amount)
        self._cash_move(-amount, payee, acct)
        self.ledger.post(acct, {acct: amount}, {"cash": amount})

    def _msg(self, sender, kind, subject, body, expires_week=None, data=None, status="info") -> Message:
        m = Message(id=self._next_id("M"), week=self.week, sender=sender, kind=kind, subject=subject,
                    body=body, expires_week=expires_week, data=data or {}, status=status)
        self.messages.append(m)
        return m

    def _event(self, kind: str, **kw) -> None:
        self.events.append({"week": self.week, "kind": kind, **kw})

    # ------------------------------------------------------------------ derived quantities
    def supplier_on_hold(self, sid: str) -> bool:
        hold = self.cfg.supplier(sid).hold_after_weeks if sid in self.sup else 99
        return any(i.supplier == sid and i.overdue_weeks(self.week) >= hold for i in self.invoices)

    def inventory_value(self) -> int:
        return sum(s.value for s in self.skus.values())

    def prepaid_total(self) -> int:
        return sum(p.prepaid for p in self.pos if p.status == "open")

    def ap_total(self) -> int:
        return sum(i.amount for i in self.invoices)

    def liabilities(self) -> int:
        return self.ap_total() + self.credit.drawn

    def current_ratio(self) -> float:
        liab = self.liabilities()
        if liab == 0:
            return float("inf")
        return (self.cash + self.inventory_value() + self.prepaid_total()) / liab

    def capacity(self) -> int:
        return self.headcount * self.cfg.staff.capacity_per_head

    def _season(self, week: int, sku_id: str) -> float:
        s = self.cfg.sku(sku_id)
        woy = self.cfg.start_week_of_year + week + s.season_phase
        return 1.0 + s.season_amp * math.sin(2 * math.pi * woy / 52.0)

    def _lambda(self, sku_id: str, week: int, ignore_shocks: bool = False) -> float:
        s = self.cfg.sku(sku_id)
        price = self.skus[sku_id].price
        ratio = max(price, 1) / self.ref_price[sku_id]
        lam = s.base_demand * self._season(week, sku_id) * ratio ** (-s.elasticity) * self.goodwill
        if not ignore_shocks:
            for sh in self.demand_shocks:
                if sh["from"] <= week <= sh["until"] and sh["sku"] in (None, sku_id):
                    lam *= sh["mult"]
        return min(lam, 25 * s.base_demand)

    def current_unit_cost(self, sid: str, qty: int) -> int:
        st = self.sup[sid]
        cost = st.unit_cost
        if st.promo and st.promo["week"] == self.week and qty >= st.promo["min_qty"]:
            cost = int(round(cost * (1 - st.promo["discount"])))
        return cost

    # ------------------------------------------------------------------ the week
    def step(self, act: dict) -> dict:
        if self.done:
            raise RuntimeError("episode is over")
        self.flows, self.events = [], []
        errors = list(act.get("_errors", []))
        rev0 = {a: self.ledger.bal[a] for a in self.ledger.bal}
        try:
            self._start_of_week()
            self._apply_action(act, errors)
            self._receive_deliveries()
            used = self._ship_bulk()
            self._retail_sales(used)
            self._shrink()
            self._weekly_costs()
            self._overdue_and_holds()
            self._covenant()
            self._update_goodwill()
            self._generate_events()
        except Bankruptcy as b:
            self.bankrupt = True
            self.bankrupt_reason = str(b)
            self.bankrupt_unpaid = b.unpaid
            self.done = True
            self._event("bankruptcy", reason=str(b))
        self.week += 1
        if self.week >= self.horizon:
            self.done = True
        from .invariants import check
        check(self)
        delta = {a: self.ledger.bal[a] - rev0[a] for a in rev0 if self.ledger.bal[a] != rev0[a]}
        return {"week": self.week - 1, "errors": errors, "flows": list(self.flows),
                "events": list(self.events), "ledger_delta": delta,
                "bankrupt": self.bankrupt, "bankrupt_reason": self.bankrupt_reason}

    def _start_of_week(self) -> None:
        t = self.week
        if self.pending_hires:
            self.headcount += self.pending_hires
            self._event("hires_joined", n=self.pending_hires)
            self.pending_hires = 0
        for sid, st in self.sup.items():
            due = [c for c in st.pending_costs if c[0] <= t]
            for w, c in due:
                sku = self.cfg.supplier(sid).sku
                self.ref_price[sku] *= 1 + 0.8 * (c / st.unit_cost - 1)  # market-wide cost inflation
                st.unit_cost = c
            st.pending_costs = [c for c in st.pending_costs if c[0] > t]
            if st.extra_lead and t > st.extra_lead_until:
                st.extra_lead = 0
            if st.promo and st.promo["week"] < t:
                st.promo = None
        for m in self.messages:
            if m.status == "open" and m.expires_week is not None and m.expires_week < t:
                m.status = "expired"
        self.demand_shocks = [s for s in self.demand_shocks if s["until"] >= t]

    # ------------------------------------------------------------------ actions
    def _apply_action(self, act: dict, errors: list[str]) -> None:
        t = self.week
        for sku, price in act.get("prices", {}).items():
            if sku not in self.skus:
                errors.append(f"price: unknown sku {sku}")
                continue
            self.skus[sku].price = price

        b = act.get("borrow", 0)
        if b > 0:
            if self.credit.frozen:
                errors.append("borrow: credit line is frozen")
            elif b > self.available_credit():
                errors.append(f"borrow: ${b / 100:,.2f} exceeds available ${self.available_credit() / 100:,.2f}")
            else:
                self._borrow(b)
        elif b < 0:
            amt = min(-b, self.credit.drawn, self.cash)
            if amt < -b:
                errors.append(f"repay: only ${amt / 100:,.2f} repaid (limited by debt or cash)")
            if amt > 0:
                self._repay(amt)

        inv_by_id = {i.id: i for i in self.invoices}
        for p in act.get("pay", []):
            inv = inv_by_id.get(p["invoice"])
            if inv is None:
                errors.append(f"pay: unknown invoice {p['invoice']}")
                continue
            want = inv.amount if p.get("amount") is None else min(p["amount"], inv.amount)
            amt = min(want, self.cash)
            if amt < want:
                errors.append(f"pay {inv.id}: insufficient cash, paid ${amt / 100:,.2f} of ${want / 100:,.2f}")
            if amt <= 0:
                continue
            inv.amount -= amt
            self._cash_move(-amt, f"supplier:{inv.supplier}", "pay_invoice")
            self.ledger.post(f"pay {inv.id}", {"ap": amt}, {"cash": amt})
        self.invoices = [i for i in self.invoices if i.amount > 0]

        inv_by_id = {i.id: i for i in self.invoices}
        for iid in act.get("dispute", []):
            inv = inv_by_id.get(iid)
            if inv is None:
                errors.append(f"dispute: unknown or settled invoice {iid}")
                continue
            if inv.overbilled > 0 and not inv.disputed:
                credit = min(inv.overbilled, inv.amount)
                inv.amount -= credit
                inv.disputed = True
                self.ledger.post(f"credit note {inv.id}", {"ap": credit}, {"overbilling": credit})
                self._msg(self._sup_name(inv.supplier), "credit_note", f"Credit note for {inv.id}",
                          f"You're right, the delivery was short. We've reduced {inv.id} by "
                          f"${credit / 100:,.2f}.")
                self._event("dispute_won", invoice=inv.id, amount=credit)
            else:
                self.counters["frivolous_disputes"] += 1
                inv.disputed = True
                errors.append(f"dispute {iid}: rejected, invoice matches the delivery")
        self.invoices = [i for i in self.invoices if i.amount > 0]

        sc = self.cfg.staff
        fire = act.get("fire", 0)
        if fire:
            fire = min(fire, self.headcount)
            cost = fire * int(round(sc.wage * sc.severance_weeks))
            if cost > self.cash:
                errors.append("fire: not enough cash for severance")
            else:
                self.headcount -= fire
                if cost:
                    self._cash_move(-cost, "staff", "severance")
                    self.ledger.post("severance", {"severance": cost}, {"cash": cost})
                self._event("fired", n=fire)
        hire = act.get("hire", 0)
        if hire:
            room = sc.max_headcount - self.headcount - self.pending_hires
            if hire > room:
                errors.append(f"hire: capped at {room} (max headcount {sc.max_headcount})")
                hire = max(0, room)
            cost = hire * sc.hire_cost
            if cost > self.cash:
                errors.append("hire: not enough cash for recruiting cost")
            elif hire:
                self.pending_hires += hire
                if cost:
                    self._cash_move(-cost, "recruiter", "hiring")
                    self.ledger.post("hiring", {"hiring": cost}, {"cash": cost})
                self._event("hired", n=hire)

        msgs = {m.id: m for m in self.messages}
        for mid in act.get("accept", []):
            m = msgs.get(mid)
            if m is None or m.status != "open":
                errors.append(f"accept: {mid} is not an open offer")
                continue
            self._accept(m, errors)

        for o in act.get("orders", []):
            self._place_order(o["supplier"], o["qty"], errors)

        for lq in act.get("liquidate", []):
            sku, qty = lq["sku"], lq["qty"]
            if sku not in self.skus:
                errors.append(f"liquidate: unknown sku {sku}")
                continue
            qty = min(qty, self.skus[sku].on_hand)
            if qty <= 0:
                continue
            frac = self.cfg.sku(sku).liquidation_frac
            cost = self._remove_fifo(sku, qty)
            proceeds = int(round(cost * frac))
            self.skus[sku].liquidated += qty
            self._cash_move(proceeds, "liquidator", "liquidation")
            self.ledger.post("liquidation", {"cash": proceeds, "liquidation_cogs": cost},
                             {"liquidation_proceeds": proceeds, "inventory": cost})
            self._event("liquidation", sku=sku, qty=qty, proceeds=proceeds, cost=cost)

        for r in act.get("replies", []):
            self.replies.append({"week": t, **r})

    def _sup_name(self, sid: str) -> str:
        try:
            return self.cfg.supplier(sid).name
        except KeyError:
            return sid

    def _place_order(self, sid: str, qty: int, errors: list[str]) -> None:
        if sid not in self.sup:
            errors.append(f"order: unknown supplier {sid}")
            return
        sc: SupplierConfig = self.cfg.supplier(sid)
        if qty < sc.min_order:
            errors.append(f"order {sid}: {qty} is below the minimum order of {sc.min_order}")
            return
        unit = self.current_unit_cost(sid, qty)
        total = unit * qty
        prepay = sc.prepay or self.supplier_on_hold(sid)
        if not prepay and sc.credit_limit is not None:
            exposure = sum(i.amount for i in self.invoices if i.supplier == sid) + sum(
                p.qty * p.unit_cost for p in self.pos if p.supplier == sid and p.status == "open" and not p.prepaid)
            if exposure + total > sc.credit_limit:
                prepay = True
                errors.append(f"order {sid}: over the supplier's credit limit of "
                              f"${sc.credit_limit / 100:,.2f}; prepayment required")
        if prepay and total > self.cash:
            errors.append(f"order {sid}: prepayment of ${total / 100:,.2f} required, "
                          f"cash is ${self.cash / 100:,.2f}")
            return
        st = self.sup[sid]
        lead = sc.lead_time + st.extra_lead
        if self.rng["supply"].random() < sc.late_delivery_p:
            lead += 1
        po = PurchaseOrder(id=self._next_id("PO"), supplier=sid, sku=sc.sku, qty=qty, unit_cost=unit,
                           placed_week=self.week, eta_week=self.week + max(1, lead))
        if prepay:
            po.prepaid = total
            self._cash_move(-total, f"supplier:{sid}", "prepay")
            self.ledger.post(f"prepay {po.id}", {"prepaid": total}, {"cash": total})
        self.pos.append(po)
        self._event("order", po=po.id, supplier=sid, sku=sc.sku, qty=qty, unit_cost=unit, prepaid=prepay)

    def _accept(self, m: Message, errors: list[str]) -> None:
        d = m.data
        if m.kind == "bulk_offer":
            m.status = "accepted"
            self.commitments.append({"msg": m.id, "sku": d["sku"], "qty": d["qty"], "price": d["price"],
                                     "deliver_week": d["deliver_week"], "customer": m.sender})
            self._event("bulk_accepted", msg=m.id, sku=d["sku"], qty=d["qty"], price=d["price"])
        elif m.kind == "vendor_offer":
            total = d["qty"] * d["unit_cost"]
            if total > self.cash:
                errors.append(f"accept {m.id}: prepayment of ${total / 100:,.2f} exceeds cash")
                return
            m.status = "accepted"
            po = PurchaseOrder(id=self._next_id("PO"), supplier=f"vendor:{m.sender}", sku=d["sku"],
                               qty=d["qty"], unit_cost=d["unit_cost"], placed_week=self.week,
                               eta_week=self.week + 1, prepaid=total, scam=not d["legit"])
            self.pos.append(po)
            self._cash_move(-total, f"vendor:{m.sender}", "prepay")
            self.ledger.post(f"prepay {po.id}", {"prepaid": total}, {"cash": total})
            self._event("vendor_offer_accepted", msg=m.id, po=po.id, total=total)
        else:
            errors.append(f"accept: {m.id} is not an offer")

    # ------------------------------------------------------------------ physical flows
    def _receive_deliveries(self) -> None:
        t = self.week
        for po in self.pos:
            if po.status != "open" or po.eta_week != t:
                continue
            st = self.skus[po.sku]
            if po.scam:
                loss = po.prepaid
                po.prepaid = 0
                po.status = "lost"
                po.cancelled = po.qty
                self.ledger.post(f"scam {po.id}", {"scam_loss": loss}, {"prepaid": loss})
                self._msg("Bank fraud desk", "notice", f"{po.id} never arrived",
                          f"The vendor behind {po.id} is unreachable and the account was closed. "
                          f"The ${loss / 100:,.2f} you prepaid is gone.")
                self._event("scam_loss", po=po.id, amount=loss)
                continue
            sc = self.cfg.supplier(po.supplier) if po.supplier in self.sup else None
            delivered = po.qty
            if sc and sc.short_ship_p > 0 and self.rng["supply"].random() < sc.short_ship_p:
                delivered = int(po.qty * self.rng["supply"].uniform(0.5, 0.9))
            short = po.qty - delivered
            po.received, po.cancelled, po.status = delivered, short, "received"
            cost_in = delivered * po.unit_cost
            if delivered:
                st.layers.append(Layer(delivered, po.unit_cost))
                st.received += delivered
            if po.prepaid > 0:
                rest = po.prepaid - cost_in
                po.prepaid = 0
                if cost_in:
                    self.ledger.post(f"receive {po.id}", {"inventory": cost_in}, {"prepaid": cost_in})
                if rest > 0:
                    if sc and sc.adversarial:
                        self.ledger.post(f"short {po.id}", {"supplier_loss": rest}, {"prepaid": rest})
                        self._event("supplier_loss", po=po.id, amount=rest)
                    else:
                        self._cash_move(rest, f"supplier:{po.supplier}", "refund")
                        self.ledger.post(f"refund {po.id}", {"cash": rest}, {"prepaid": rest})
            else:
                overbill = 0
                if short and sc and self.rng["supply"].random() < sc.overbill_p:
                    overbill = short * po.unit_cost
                amount = cost_in + overbill
                if amount:
                    inv = Invoice(id=self._next_id("INV"), supplier=po.supplier, po=po.id, amount=amount,
                                  original=amount, issued_week=t, due_week=t + sc.terms_weeks,
                                  overbilled=overbill)
                    self.invoices.append(inv)
                    debits = {"inventory": cost_in} if cost_in else {}
                    if overbill:
                        debits["overbilling"] = overbill
                    self.ledger.post(f"invoice {inv.id}", debits, {"ap": amount})
            name = self._sup_name(po.supplier)
            body = f"{delivered} units of {po.sku} delivered against {po.id}."
            if short:
                body += f" {short} units could not be supplied."
            self._msg(name, "delivery", f"Delivery {po.id}", body)
            self._event("delivery", po=po.id, sku=po.sku, qty=delivered, short=short)

    def _ship_bulk(self) -> int:
        t = self.week
        used = 0
        cap = self.capacity()
        keep = []
        for c in self.commitments:
            if c["deliver_week"] != t:
                keep.append(c)
                continue
            st = self.skus[c["sku"]]
            q = min(c["qty"], st.on_hand, cap - used)
            if q > 0:
                cost = self._remove_fifo(c["sku"], q)
                rev = q * c["price"]
                st.sold_bulk += q
                used += q
                self._cash_move(rev, f"customer:{c['customer']}", "bulk_sale")
                self.ledger.post("bulk sale", {"cash": rev, "cogs": cost}, {"sales_bulk": rev, "inventory": cost})
            if q < c["qty"]:
                self.goodwill = max(0.3, self.goodwill - 0.04)
                self._msg(c["customer"], "notice", "Order short",
                          f"We received {q} of the {c['qty']} units of {c['sku']} we agreed. Disappointing.")
            self._event("bulk_shipped", sku=c["sku"], qty=q, ordered=c["qty"], revenue=q * c["price"])
        self.commitments = keep
        return used

    def _retail_sales(self, cap_used: int) -> None:
        t = self.week
        cap = max(0, self.capacity() - cap_used)
        ids = list(self.skus)
        demand = {k: poisson(self._cell_rng("demand", t, k), self._lambda(k, t)) for k in ids}
        avail = {k: min(demand[k], self.skus[k].on_hand) for k in ids}
        total = sum(avail.values())
        sold = dict(avail)
        if total > cap:
            exact = {k: avail[k] * cap / total for k in ids}
            sold = {k: int(math.floor(exact[k])) for k in ids}
            rem = cap - sum(sold.values())
            for k in sorted(ids, key=lambda k: (-(exact[k] - sold[k]), ids.index(k)))[:rem]:
                sold[k] += 1
        self._fill = (sum(sold.values()) / sum(demand.values())) if sum(demand.values()) else 1.0
        for k in ids:
            st = self.skus[k]
            q = sold[k]
            st.demand_last, st.sold_last = demand[k], q
            st.stockout_last = q < demand[k]
            if st.stockout_last:
                self.counters["stockout_weeks"] += 1
                self.counters["lost_units"] += demand[k] - q
            st.sales_history.append(q)
            if len(st.sales_history) > 12:
                st.sales_history.pop(0)
            if q <= 0:
                continue
            cost = self._remove_fifo(k, q)
            rev = q * st.price
            st.sold_retail += q
            self._cash_move(rev, "customers", "retail_sale")
            self.ledger.post(f"retail {k}", {"cash": rev, "cogs": cost}, {"sales_retail": rev, "inventory": cost})
        self._event("retail", demand=demand, sold=sold, capacity=cap)

    def _shrink(self) -> None:
        for k, st in self.skus.items():
            rate = self.cfg.sku(k).shrink_rate
            q = min(st.on_hand, stoch_round(self._cell_rng("shrink", self.week, k), st.on_hand * rate))
            if q > 0:
                cost = self._remove_fifo(k, q)
                st.shrunk += q
                self.ledger.post(f"shrink {k}", {"shrink": cost}, {"inventory": cost})

    def _weekly_costs(self) -> None:
        cfg = self.cfg
        self._pay_mandatory(self.headcount * cfg.staff.wage, "staff", "wages")
        self._pay_mandatory(cfg.rent, "landlord", "rent")
        hold = sum(st.on_hand * cfg.sku(k).holding_cost for k, st in self.skus.items())
        self._pay_mandatory(hold, "warehouse", "holding")
        apr = cfg.credit.apr + (cfg.credit.penalty_apr if self.credit.in_breach else 0.0)
        self._pay_mandatory(int(round(self.credit.drawn * apr / 52)), "bank", "interest")
        cd = self.credit.called_due_week
        if cd is not None and self.week >= cd and self.credit.drawn > 0:
            owed = self.credit.drawn
            if owed > self.cash:
                raise Bankruptcy(f"credit line called; ${owed / 100:,.2f} due, cash ${self.cash / 100:,.2f}")
            self._repay(owed)
            self._event("loan_repaid_on_call", amount=owed)

    def _overdue_and_holds(self) -> None:
        t = self.week
        for inv in self.invoices:
            od = inv.overdue_weeks(t)
            if od <= 0:
                continue
            sc = self.cfg.supplier(inv.supplier) if inv.supplier in self.sup else None
            rate = sc.late_fee_rate if sc else 0.02
            fee = int(round(inv.amount * rate))
            if fee:
                inv.amount += fee
                inv.late_fees += fee
                self.counters["late_fee_total"] += fee
                self.ledger.post(f"late fee {inv.id}", {"late_fees": fee}, {"ap": fee})
            name = self._sup_name(inv.supplier)
            if od == 1:
                self._msg(name, "reminder", f"{inv.id} is overdue",
                          f"Invoice {inv.id} (${inv.amount / 100:,.2f}) was due in week {inv.due_week}. "
                          f"A late fee of {rate:.0%} per week now applies.")
            if sc and od == sc.hold_after_weeks:
                self._msg(name, "credit_hold", "Account on credit hold",
                          f"{inv.id} is {od} weeks overdue. Until it is paid, we need payment in full "
                          f"when you place an order.")
                self._event("credit_hold", supplier=inv.supplier)

    def _covenant(self) -> None:
        cc = self.cfg.credit
        if (self.week + 1) % cc.test_every_weeks != 0:
            return
        ratio = self.current_ratio()
        self.credit.last_ratio = ratio
        ok = ratio >= cc.covenant_min_current_ratio
        if ok:
            if self.credit.in_breach:
                self._msg("Bank", "covenant", "Covenant cured",
                          f"Current ratio {ratio:.2f} meets the {cc.covenant_min_current_ratio:.2f} minimum. "
                          "Penalty rate removed.")
            self.credit.in_breach = False
            self.credit.breach_streak = 0
            self._event("covenant_pass", ratio=ratio)
            return
        self.credit.in_breach = True
        self.credit.breach_streak += 1
        self.credit.breaches += 1
        self._event("covenant_breach", ratio=ratio, streak=self.credit.breach_streak)
        if self.credit.breach_streak >= 2 and self.credit.called_due_week is None and self.credit.drawn > 0:
            self.credit.frozen = True
            self.credit.called_due_week = self.week + cc.call_grace_weeks
            self._msg("Bank", "covenant", "Credit line called",
                      f"Second consecutive covenant breach (current ratio {ratio:.2f}). The line is frozen "
                      f"and the full balance of ${self.credit.drawn / 100:,.2f} is due in week "
                      f"{self.credit.called_due_week}.")
            self._event("line_called", due=self.credit.called_due_week)
        else:
            self._msg("Bank", "covenant", "Covenant breach",
                      f"Your current ratio is {ratio:.2f}, below the {cc.covenant_min_current_ratio:.2f} "
                      f"minimum. A penalty rate of +{cc.penalty_apr:.0%} applies until cured. A second "
                      "consecutive breach freezes and calls the line.")

    def _update_goodwill(self) -> None:
        fill = getattr(self, "_fill", 1.0)
        ratios = [self.skus[k].price / self.ref_price[k] for k in self.skus]
        avg = sum(ratios) / len(ratios)
        g = self.goodwill
        delta = 0.04 * (1.0 - g)
        if fill < 0.97:
            delta -= 0.25 * (0.97 - fill)
        if avg > 1.15:
            delta -= 0.15 * (avg - 1.15)
        elif avg < 0.95:
            delta += 0.02 * (0.95 - avg)
        self.goodwill = min(1.25, max(0.3, g + delta))

    # ------------------------------------------------------------------ events for next week
    def _generate_events(self) -> None:
        cfg, ev, r = self.cfg, self.cfg.events, self.rng["events"]
        nxt = self.week + 1
        sku_ids = [s.id for s in cfg.skus]

        if r.random() < ev.bulk_offer_p:
            s = cfg.sku(r.choice(sku_ids))
            qty = max(10, int(s.base_demand * r.uniform(0.3, 0.8)))
            price = int(self.ref_price[s.id] * r.uniform(0.6, 1.1))
            buyer = r.choice(["Harbor Catering", "Northside School", "Pine St Cafe", "City Events Co",
                              "Greenway Hotel"])
            self._msg(buyer, "bulk_offer", f"Bulk order request: {qty} x {s.name}",
                      f"We'd like {qty} units of {s.name} ({s.id}) at ${price / 100:,.2f} each, delivered in "
                      f"week {nxt + 1}, paid on delivery. Accept this week if you can do it.",
                      expires_week=nxt, status="open",
                      data={"sku": s.id, "qty": qty, "price": price, "deliver_week": nxt + 1})

        if cfg.suppliers and r.random() < ev.promo_p:
            sc = r.choice(cfg.suppliers)
            disc = round(r.uniform(0.08, 0.15), 2)
            self.sup[sc.id].promo = {"week": nxt, "min_qty": 2 * sc.min_order, "discount": disc}
            self._msg(sc.name, "promo", f"{int(disc * 100)}% off this week",
                      f"Orders of {2 * sc.min_order}+ units of {sc.sku} placed in week {nxt} get "
                      f"{int(disc * 100)}% off.")

        if cfg.suppliers and r.random() < ev.cost_increase_p:
            sc = r.choice(cfg.suppliers)
            st = self.sup[sc.id]
            new = int(st.unit_cost * r.uniform(1.05, 1.2))
            st.pending_costs.append((nxt + 2, new))
            self._msg(sc.name, "price_change", "Price increase",
                      f"From week {nxt + 2}, {sc.sku} will cost ${new / 100:,.2f} per unit "
                      f"(was ${st.unit_cost / 100:,.2f}).")
            self._event("cost_increase", supplier=sc.id, new=new)

        if cfg.suppliers and r.random() < ev.supplier_disruption_p:
            sc = r.choice(cfg.suppliers)
            st = self.sup[sc.id]
            extra = r.randint(1, 2)
            st.extra_lead, st.extra_lead_until = extra, nxt + r.randint(2, 4)
            delayed = 0
            for po in self.pos:
                if po.status == "open" and po.supplier == sc.id and po.eta_week >= nxt:
                    po.eta_week += extra
                    delayed += 1
            self._msg(sc.name, "disruption", "Shipping delays",
                      f"A problem at our warehouse adds {extra} week(s) to deliveries until week "
                      f"{st.extra_lead_until}. {delayed} of your open orders are affected.")
            self._event("supplier_disruption", supplier=sc.id, extra=extra)

        if r.random() < ev.demand_shock_p:
            sku = r.choice(sku_ids + [None])
            up = r.random() < 0.4
            mult = r.uniform(1.25, 1.6) if up else r.uniform(0.5, 0.75)
            dur = r.randint(2, 6)
            self.demand_shocks.append({"sku": sku, "mult": mult, "from": nxt, "until": nxt + dur - 1})
            self._event("demand_shock", sku=sku, mult=round(mult, 3), weeks=dur)
            if r.random() < 0.6:
                what = cfg.sku(sku).name if sku else "the whole shop"
                hint = (f"A local event is expected to lift demand for {what} for a few weeks." if up else
                        f"Road works on Main St are expected to cut foot traffic for {what} for a few weeks.")
                self._msg("Town Gazette", "news", "Local news", hint)

        if r.random() < ev.scam_p:
            s = cfg.sku(r.choice(sku_ids))
            qty = max(20, int(s.base_demand * r.uniform(1.0, 3.0)))
            unit = int(s.start_unit_cost * r.uniform(0.45, 0.6))
            legit = r.random() < ev.scam_legit_p
            vendor = r.choice(["QuickStock Wholesale", "BestDeal Imports", "Overstock Direct LLC"])
            self._msg(vendor, "vendor_offer", f"Clearance: {s.name} at {int(100 * unit / s.start_unit_cost)}% of list",
                      f"New vendor here. {qty} units of {s.name} ({s.id}) at ${unit / 100:,.2f}/unit, "
                      f"delivered next week. Payment in full up front to our account. Offer ends this week.",
                      expires_week=nxt, status="open",
                      data={"sku": s.id, "qty": qty, "unit_cost": unit, "legit": legit})
