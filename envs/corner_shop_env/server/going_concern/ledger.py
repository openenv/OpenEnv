"""Double-entry ledger in integer cents. Every economic event posts a balanced entry.
The ledger is kept independently of the operational state so the invariant checker
can reconcile one against the other."""
from __future__ import annotations

ASSETS = ("cash", "inventory", "prepaid")
LIABILITIES = ("ap", "debt")
EQUITY = ("capital",)
REVENUE = ("sales_retail", "sales_bulk", "liquidation_proceeds")
EXPENSES = (
    "cogs", "liquidation_cogs", "wages", "rent", "interest", "holding", "late_fees",
    "severance", "hiring", "shrink", "supplier_loss", "overbilling", "scam_loss",
)
DEBIT_NORMAL = set(ASSETS) | set(EXPENSES)
ALL_ACCOUNTS = ASSETS + LIABILITIES + EQUITY + REVENUE + EXPENSES


class InvariantViolation(Exception):
    pass


class Ledger:
    def __init__(self):
        self.bal: dict[str, int] = {a: 0 for a in ALL_ACCOUNTS}
        self.n_entries = 0

    def post(self, memo: str, debits: dict[str, int], credits: dict[str, int]) -> None:
        for side in (debits, credits):
            for acct, amt in side.items():
                if acct not in self.bal:
                    raise InvariantViolation(f"unknown account {acct!r} in {memo!r}")
                if not isinstance(amt, int):
                    raise InvariantViolation(f"non-integer amount {amt!r} in {memo!r}")
        if sum(debits.values()) != sum(credits.values()):
            raise InvariantViolation(f"unbalanced entry {memo!r}: {debits} vs {credits}")
        for acct, amt in debits.items():
            self.bal[acct] += amt if acct in DEBIT_NORMAL else -amt
        for acct, amt in credits.items():
            self.bal[acct] += -amt if acct in DEBIT_NORMAL else amt
        self.n_entries += 1

    def total(self, accts) -> int:
        return sum(self.bal[a] for a in accts)

    def equity(self) -> int:
        return self.bal["capital"] + self.total(REVENUE) - self.total(EXPENSES)

    def balances(self) -> bool:
        return self.total(ASSETS) == self.total(LIABILITIES) + self.equity()
