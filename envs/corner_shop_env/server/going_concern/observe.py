"""What the agent sees: structured state plus inbox text. Goodwill, true demand and
upcoming shocks stay hidden."""
from __future__ import annotations


def d(cents: int) -> float:
    return round(cents / 100, 2)


def observe(sim, errors: list[str] | None = None) -> dict:
    cfg, t = sim.cfg, sim.week
    skus = []
    for s in cfg.skus:
        st = sim.skus[s.id]
        in_transit = sum(p.qty for p in sim.pos if p.status == "open" and p.sku == s.id)
        oh = st.on_hand
        skus.append({
            "id": s.id, "name": s.name, "price": d(st.price), "on_hand": oh,
            "avg_unit_cost": d(st.value // oh) if oh else None,
            "in_transit": in_transit, "sales_history": list(st.sales_history[-8:]),
            "stockout_last_week": st.stockout_last, "holding_cost_per_unit_week": d(s.holding_cost),
        })
    suppliers = []
    for s in cfg.suppliers:
        ss = sim.sup[s.id]
        suppliers.append({
            "id": s.id, "name": s.name, "sku": s.sku, "unit_cost": d(ss.unit_cost),
            "lead_time_weeks": s.lead_time + ss.extra_lead, "min_order": s.min_order,
            "terms_weeks": s.terms_weeks,
            "requires_prepayment": s.prepay or sim.supplier_on_hold(s.id),
            "on_credit_hold": sim.supplier_on_hold(s.id),
            "credit_limit": d(s.credit_limit) if s.credit_limit is not None else None,
            "owed_plus_on_order": d(sum(i.amount for i in sim.invoices if i.supplier == s.id) + sum(
                p.qty * p.unit_cost for p in sim.pos if p.supplier == s.id and p.status == "open" and not p.prepaid)),
        })
    inbox = []
    for m in sim.messages:
        if m.week >= t - 1 or m.status == "open":
            if m.status == "expired":
                continue
            inbox.append({"id": m.id, "week": m.week, "from": m.sender, "kind": m.kind,
                          "subject": m.subject, "body": m.body,
                          "actionable": m.status == "open",
                          **({"expires_week": m.expires_week} if m.expires_week is not None else {})})
    cc = sim.cfg.credit
    nxt = t + (cc.test_every_weeks - 1 - t % cc.test_every_weeks)
    ratio = sim.current_ratio()
    return {
        "firm": cfg.name, "week": t, "weeks_total": sim.horizon, "done": sim.done,
        "bankrupt": sim.bankrupt,
        "cash": d(sim.cash),
        "credit_line": {
            "limit": d(cc.limit), "drawn": d(sim.credit.drawn), "available": d(sim.available_credit()),
            "apr": cc.apr + (cc.penalty_apr if sim.credit.in_breach else 0.0),
            "frozen": sim.credit.frozen, "called_due_week": sim.credit.called_due_week,
            "covenant": {"type": "min_current_ratio", "minimum": cc.covenant_min_current_ratio,
                         "current_ratio": None if ratio == float("inf") else round(ratio, 3),
                         "tested_end_of_week": nxt, "in_breach": sim.credit.in_breach},
        },
        "staff": {"headcount": sim.headcount, "pending_hires": sim.pending_hires,
                  "wage_per_week": d(cfg.staff.wage), "capacity_units_per_week": sim.capacity(),
                  "capacity_per_head": cfg.staff.capacity_per_head, "hire_cost": d(cfg.staff.hire_cost),
                  "severance_weeks": cfg.staff.severance_weeks, "max_headcount": cfg.staff.max_headcount},
        "fixed_costs_per_week": {"rent": d(cfg.rent)},
        "skus": skus,
        "suppliers": suppliers,
        "purchase_orders": [{"id": p.id, "supplier": p.supplier, "sku": p.sku, "qty": p.qty,
                             "unit_cost": d(p.unit_cost), "eta_week": p.eta_week, "prepaid": d(p.prepaid)}
                            for p in sim.pos if p.status == "open"],
        "invoices": [{"id": i.id, "supplier": i.supplier, "po": i.po, "amount": d(i.amount), "due_week": i.due_week,
                      "overdue_weeks": i.overdue_weeks(t)} for i in sim.invoices],
        "commitments": [{"sku": c["sku"], "qty": c["qty"], "price": d(c["price"]),
                         "deliver_week": c["deliver_week"], "customer": c["customer"]}
                        for c in sim.commitments],
        "inbox": inbox,
        "errors": list(errors or []),
    }


def render_text(obs: dict) -> str:
    """A compact text view for LLM agents."""
    o = obs
    cl = o["credit_line"]
    lines = [
        f"# {o['firm']} — week {o['week']} of {o['weeks_total']}",
        f"Cash ${o['cash']:,.2f} | credit drawn ${cl['drawn']:,.2f} of ${cl['limit']:,.2f} "
        f"(available ${cl['available']:,.2f}, APR {cl['apr']:.1%}{', FROZEN' if cl['frozen'] else ''})",
        f"Covenant: current ratio {cl['covenant']['current_ratio']} vs minimum {cl['covenant']['minimum']}, "
        f"next test end of week {cl['covenant']['tested_end_of_week']}"
        + (f"; LINE CALLED, due week {cl['called_due_week']}" if cl["called_due_week"] is not None else ""),
        f"Staff {o['staff']['headcount']} (+{o['staff']['pending_hires']} joining), "
        f"${o['staff']['wage_per_week']:,.2f}/week each, capacity {o['staff']['capacity_units_per_week']} units/week. "
        f"Rent ${o['fixed_costs_per_week']['rent']:,.2f}/week.",
        "## Products",
    ]
    for s in o["skus"]:
        lines.append(f"- {s['id']} {s['name']}: price ${s['price']:.2f}, on hand {s['on_hand']}"
                     f" (avg cost {s['avg_unit_cost']}), in transit {s['in_transit']}, "
                     f"recent sales {s['sales_history']}{', STOCKED OUT last week' if s['stockout_last_week'] else ''}")
    lines.append("## Suppliers")
    for s in o["suppliers"]:
        lines.append(f"- {s['id']} {s['name']} sells {s['sku']} at ${s['unit_cost']:.2f}, lead {s['lead_time_weeks']}w, "
                     f"min {s['min_order']}, net {s['terms_weeks']}w"
                     f"{', PREPAY' if s['requires_prepayment'] else ''}{', ON HOLD' if s['on_credit_hold'] else ''}")
    if o["purchase_orders"]:
        lines.append("## Open purchase orders")
        lines += [f"- {p['id']} {p['qty']} x {p['sku']} from {p['supplier']}, ETA week {p['eta_week']}"
                  for p in o["purchase_orders"]]
    if o["invoices"]:
        lines.append("## Invoices owed")
        lines += [f"- {i['id']} {i['supplier']} ${i['amount']:,.2f} due week {i['due_week']}"
                  + (f" (OVERDUE {i['overdue_weeks']}w)" if i["overdue_weeks"] else "") for i in o["invoices"]]
    if o["commitments"]:
        lines.append("## Customer orders to ship")
        lines += [f"- {c['qty']} x {c['sku']} at ${c['price']:.2f} to {c['customer']} in week {c['deliver_week']}"
                  for c in o["commitments"]]
    lines.append("## Inbox")
    if not o["inbox"]:
        lines.append("(empty)")
    for m in o["inbox"]:
        tag = " [OFFER, accept by id]" if m["actionable"] else ""
        lines.append(f"- {m['id']} from {m['from']}: {m['subject']}{tag}\n  {m['body']}")
    if o["errors"]:
        lines.append("## Problems with your last action")
        lines += [f"- {e}" for e in o["errors"]]
    return "\n".join(lines)


def render_compact(obs: dict) -> str:
    """Token-lean view for training: same decision-relevant facts as render_text, about 3x fewer tokens.
    Multi-turn episodes resend every past observation, so this is the main cost lever."""
    o, cl = obs, obs["credit_line"]
    cov = cl["covenant"]
    m = lambda x: f"${x:,.0f}"
    lines = [
        f"Week {o['week']}/{o['weeks_total']} | Cash {m(o['cash'])} | Line {m(cl['drawn'])}/{m(cl['limit'])} "
        f"{cl['apr']:.1%}{' FROZEN' if cl['frozen'] else ''}"
        + (f" CALLED due wk {cl['called_due_week']}" if cl["called_due_week"] is not None else "")
        + f" | Covenant {cov['current_ratio']} min {cov['minimum']} test wk {cov['tested_end_of_week']}",
        f"Staff {o['staff']['headcount']}{'+' + str(o['staff']['pending_hires']) if o['staff']['pending_hires'] else ''} "
        f"({m(o['staff']['wage_per_week'])}/wk, cap {o['staff']['capacity_units_per_week']}/wk) | Rent {m(o['fixed_costs_per_week']['rent'])}/wk",
        "Products: " + " | ".join(
            f"{s['id']} {s['name']} ${s['price']:.2f} stock {s['on_hand']}"
            + (f" +{s['in_transit']} coming" if s["in_transit"] else "")
            + f" sold {','.join(map(str, s['sales_history'][-4:]))}" + (" STOCKOUT" if s["stockout_last_week"] else "")
            for s in o["skus"]),
        "Suppliers: " + " | ".join(
            f"{s['id']}>{s['sku']} ${s['unit_cost']:.2f} {s['lead_time_weeks']}w min {s['min_order']} net {s['terms_weeks']}w"
            + (f" credit left ${max(0, s['credit_limit'] - s['owed_plus_on_order']):,.0f}" if s.get("credit_limit") else "")
            + (" PREPAY" if s["requires_prepayment"] else "") + (" HOLD" if s["on_credit_hold"] else "")
            for s in o["suppliers"]),
    ]
    if o["invoices"]:
        lines.append("Owed: " + " | ".join(
            f"{i['id']} {i['supplier']} {m(i['amount'])} due wk {i['due_week']}"
            + (f" OVERDUE {i['overdue_weeks']}w" if i["overdue_weeks"] else "") for i in o["invoices"]))
    if o["purchase_orders"]:
        by = {}
        for p in o["purchase_orders"]:
            by.setdefault(p["sku"], {}).setdefault(p["eta_week"], 0)
            by[p["sku"]][p["eta_week"]] += p["qty"]
        lines.append("Arriving (qty@week): " + " | ".join(
            f"{k} " + ",".join(f"{q}@{w}" for w, q in sorted(v.items())) for k, v in sorted(by.items())))
    if o["commitments"]:
        lines.append("To ship: " + " | ".join(f"{c['qty']} {c['sku']} @${c['price']:.2f} wk {c['deliver_week']}" for c in o["commitments"]))
    msgs = [x for x in o["inbox"] if x["kind"] != "delivery"]  # deliveries already show in stock/arrivals
    if msgs:
        lines.append("Inbox: " + " | ".join(
            f"{x['id']}{' [OFFER]' if x['actionable'] else ''} {x['from']}: {x['body']}" for x in msgs))
    if o["errors"]:
        lines.append("Problems last week: " + "; ".join(e.replace("prepayment required", "prepay needed") for e in o["errors"]))
    return "\n".join(lines)
