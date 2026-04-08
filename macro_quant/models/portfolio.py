"""
models/portfolio.py
Position, Portfolio, PortfolioMetrics — bond portfolio management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from macro_quant.models.instruments import Bond, BondETF, BondFuture, Instrument


# ──────────────────────────────────────────────
# Position
# ──────────────────────────────────────────────

class Position(BaseModel):
    """A portfolio line item: instrument + quantity + cost."""

    instrument: Bond | BondETF | BondFuture
    quantity: float = Field(gt=0, description="Number of contracts / bonds / shares")
    avg_cost: float = Field(gt=0, description="Average purchase price")
    opened_at: datetime = Field(default_factory=datetime.now)
    notes: str = ""

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def market_value(self) -> float | None:
        """Current market value (None if no price available)."""
        price = getattr(self.instrument, "price", None)
        if price is None:
            return None
        # Bonds: price as % of face value
        if isinstance(self.instrument, Bond):
            return self.quantity * self.instrument.face_value * (price / 100)
        return self.quantity * price

    @computed_field
    @property
    def cost_basis(self) -> float:
        """Total purchase value."""
        if isinstance(self.instrument, Bond):
            return self.quantity * self.instrument.face_value * (self.avg_cost / 100)
        return self.quantity * self.avg_cost

    @computed_field
    @property
    def unrealized_pnl(self) -> float | None:
        mv = self.market_value
        if mv is None:
            return None
        return round(mv - self.cost_basis, 2)

    @computed_field
    @property
    def unrealized_pnl_pct(self) -> float | None:
        pnl = self.unrealized_pnl
        if pnl is None:
            return None
        return round(pnl / self.cost_basis, 6)

    @computed_field
    @property
    def dv01(self) -> float | None:
        """Position DV01 (sensitivity to +1bp rate move)."""
        instr = self.instrument
        if isinstance(instr, Bond) and instr.modified_duration and instr.price:
            mv = self.market_value
            if mv:
                return round(-mv * instr.modified_duration / 10_000, 2)
        if isinstance(instr, BondETF) and instr.avg_duration and self.market_value:
            return round(-self.market_value * instr.avg_duration / 10_000, 2)
        if isinstance(instr, BondFuture) and instr.dv01:
            return instr.dv01 * self.quantity
        return None

    @computed_field
    @property
    def spread_dv01(self) -> float | None:
        """Position Spread DV01 (sensitivity to +1bp spread move)."""
        instr = self.instrument
        if isinstance(instr, BondETF) and instr.spread_duration is not None and self.market_value:
            return round(-self.market_value * instr.spread_duration / 10_000, 2)
        return None

    def describe(self) -> str:
        pnl_str = f"PnL: {self.unrealized_pnl_pct:.2%}" if self.unrealized_pnl_pct else ""
        mv_str = f"MV: ${self.market_value:,.0f}" if self.market_value else ""
        return f"{self.instrument.ticker} x{self.quantity} | {mv_str} | {pnl_str}"


# ──────────────────────────────────────────────
# PortfolioMetrics — computed aggregates
# ──────────────────────────────────────────────

class PortfolioMetrics(BaseModel):
    """Snapshot of the portfolio's aggregated metrics."""

    as_of: datetime = Field(default_factory=datetime.now)
    total_market_value: float = 0.0
    total_cost_basis: float = 0.0
    total_unrealized_pnl: float = 0.0

    # Aggregated sensitivities (weighted average)
    weighted_avg_duration: float | None = None
    weighted_avg_ytm: float | None = None
    weighted_avg_maturity: float | None = None
    portfolio_dv01: float | None = None       # Total DV01 in $

    # Concentration
    largest_position_pct: float | None = None
    n_positions: int = 0

    @computed_field
    @property
    def total_return_pct(self) -> float | None:
        if self.total_cost_basis > 0:
            return round(self.total_unrealized_pnl / self.total_cost_basis, 6)
        return None

    def describe(self) -> str:
        lines = [
            f"── Portfolio Metrics [{self.as_of.strftime('%Y-%m-%d %H:%M')}] ──",
            f"Market Value : ${self.total_market_value:>15,.0f}",
            f"Cost Basis   : ${self.total_cost_basis:>15,.0f}",
            f"Unrealized   : ${self.total_unrealized_pnl:>+15,.0f}  ({self.total_return_pct:.2%})"
            if self.total_return_pct else f"Unrealized   : ${self.total_unrealized_pnl:>+15,.0f}",
        ]
        if self.weighted_avg_duration:
            lines.append(f"Avg Duration : {self.weighted_avg_duration:.2f} years")
        if self.weighted_avg_ytm:
            lines.append(f"Avg YTM      : {self.weighted_avg_ytm:.3%}")
        if self.portfolio_dv01:
            lines.append(f"DV01 total   : ${self.portfolio_dv01:,.0f}")
        if self.weighted_avg_maturity:
            lines.append(f"Avg Maturity : {self.weighted_avg_maturity:.1f} years")
        lines.append(f"Positions    : {self.n_positions}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Portfolio
# ──────────────────────────────────────────────

class Portfolio(BaseModel):
    """
    Bond / macro portfolio.
    Manages positions, computes aggregated metrics.
    """

    name: str
    description: str = ""
    positions: dict[str, Position] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    benchmark: str | None = None     # ex: "AGG", "TLT"
    currency: str = "USD"
    cash: float = 0.0

    # ── Position management ──

    def add_position(self, position: Position) -> Portfolio:
        """Returns a new Portfolio with the position added (immutable)."""
        ticker = position.instrument.ticker
        new_positions = dict(self.positions)

        if ticker in new_positions:
            # Weighted average if existing position
            existing = new_positions[ticker]
            total_qty = existing.quantity + position.quantity
            avg_cost = (
                existing.avg_cost * existing.quantity + position.avg_cost * position.quantity
            ) / total_qty
            new_positions[ticker] = Position(
                instrument=position.instrument,
                quantity=total_qty,
                avg_cost=avg_cost,
                opened_at=existing.opened_at,
                notes=existing.notes,
            )
        else:
            new_positions[ticker] = position

        return self.model_copy(update={"positions": new_positions})

    def remove_position(self, ticker: str) -> Portfolio:
        new_positions = {k: v for k, v in self.positions.items() if k != ticker}
        return self.model_copy(update={"positions": new_positions})

    def update_prices(self, prices: dict[str, float]) -> Portfolio:
        """Updates market prices and returns a new Portfolio."""
        new_positions = {}
        for ticker, pos in self.positions.items():
            if ticker in prices:
                updated_instrument = pos.instrument.model_copy(
                    update={"price": prices[ticker]}
                )
                new_positions[ticker] = pos.model_copy(
                    update={"instrument": updated_instrument}
                )
            else:
                new_positions[ticker] = pos
        return self.model_copy(update={"positions": new_positions})

    # ── Metrics ──

    def compute_metrics(self) -> PortfolioMetrics:
        """Computes the portfolio's aggregated metrics."""
        total_mv = 0.0
        total_cost = 0.0
        total_pnl = 0.0
        total_dv01 = 0.0
        dur_weighted = 0.0
        ytm_weighted = 0.0
        mat_weighted = 0.0
        max_weight = 0.0

        positions_with_mv = []
        for pos in self.positions.values():
            mv = pos.market_value
            if mv is not None:
                total_mv += mv
                positions_with_mv.append((pos, mv))

            total_cost += pos.cost_basis
            pnl = pos.unrealized_pnl
            if pnl is not None:
                total_pnl += pnl

            dv01 = pos.dv01
            if dv01 is not None:
                total_dv01 += dv01

        # Weighted averages (as % of MV)
        if total_mv > 0:
            for pos, mv in positions_with_mv:
                w = mv / total_mv
                max_weight = max(max_weight, w)
                instr = pos.instrument
                if isinstance(instr, Bond):
                    if instr.modified_duration:
                        dur_weighted += w * instr.modified_duration
                    if instr.ytm:
                        ytm_weighted += w * instr.ytm
                    if instr.years_to_maturity:
                        mat_weighted += w * instr.years_to_maturity
                elif isinstance(instr, BondETF):
                    if instr.avg_duration:
                        dur_weighted += w * instr.avg_duration
                    if instr.avg_ytm:
                        ytm_weighted += w * instr.avg_ytm
                    if instr.avg_maturity:
                        mat_weighted += w * instr.avg_maturity

        return PortfolioMetrics(
            total_market_value=total_mv,
            total_cost_basis=total_cost,
            total_unrealized_pnl=total_pnl,
            weighted_avg_duration=round(dur_weighted, 3) if dur_weighted else None,
            weighted_avg_ytm=round(ytm_weighted, 6) if ytm_weighted else None,
            weighted_avg_maturity=round(mat_weighted, 2) if mat_weighted else None,
            portfolio_dv01=round(total_dv01, 2) if total_dv01 else None,
            largest_position_pct=round(max_weight, 4) if max_weight else None,
            n_positions=len(self.positions),
        )

    # ── Iteration ──

    def __iter__(self) -> Iterator[Position]:
        return iter(self.positions.values())

    def __len__(self) -> int:
        return len(self.positions)

    def __contains__(self, ticker: str) -> bool:
        return ticker in self.positions

    # ── Allocations ──

    def allocation(self) -> dict[str, float]:
        """Returns the weight of each position as % of total MV."""
        metrics = self.compute_metrics()
        if metrics.total_market_value == 0:
            return {}
        return {
            ticker: round((pos.market_value or 0) / metrics.total_market_value, 4)
            for ticker, pos in self.positions.items()
        }

    def describe(self) -> str:
        metrics = self.compute_metrics()
        lines = [f"═══ PORTFOLIO: {self.name} ═══", ""]
        for ticker, pos in self.positions.items():
            lines.append(f"  • {pos.describe()}")
        lines.append("")
        lines.append(metrics.describe())
        return "\n".join(lines)
