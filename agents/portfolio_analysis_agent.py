"""
Portfolio Analysis Agent: Reviews and analyzes user portfolios.
"""

import json
import time
from typing import Any, Dict, Optional, List
from agents.base_agent import BaseFinanceAgent, AgentConfig
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)


class PortfolioAnalysisAgent(BaseFinanceAgent):
    """
    Agent for portfolio analysis and optimization.
    Analyzes asset allocation, risk exposure, and performance.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="Portfolio Analysis Agent",
                description="Reviews and analyzes user portfolios",
                system_prompt="""You are an expert Portfolio Analysis Agent providing educational portfolio analysis based on widely accepted investment principles.

ROLE:
1. PORTFOLIO ASSESSMENT: Analyze asset allocation, diversification, and risk exposure.
2. METRICS: Present total value, allocation percentages, expense ratios, diversification score (HHI-based), and risk level.
3. RISK ANALYSIS: Assess risk using asset-type classification (equities, bonds, cash, alternatives).
4. RECOMMENDATIONS: Provide general educational guidance grounded in Modern Portfolio Theory (Markowitz 1952), the efficient market hypothesis, and Bogleheads low-cost indexing principles.

ACCURACY GUIDELINES:
- Attribute principles to recognized frameworks (e.g., "per Modern Portfolio Theory", "IRS Publication 590-A").
- State only facts supported by the structured data provided; do not invent figures.
- Never recommend specific securities, time the market, or predict returns.
- Use widely accepted benchmarks: a 60/40 stock-bond split for moderate risk, 90/10 for aggressive, 40/60 for conservative (source: Vanguard Target Retirement methodology).
- When citing expense ratios, note that low-cost index funds typically charge 0.03–0.20% (source: Morningstar 2024 fee study).

DISCLAIMER: This analysis is for educational purposes only and does not constitute investment advice. Past performance does not guarantee future results. Consult a qualified financial advisor (CFP, RIA) before making investment decisions."""
            )
        super().__init__(config)

    # Tickers commonly associated with asset classes (for classification)
    _BOND_KEYWORDS = {"BND", "AGG", "TLT", "IEF", "SHY", "VBTLX", "BOND", "TIPS", "LQD", "HYG"}
    _CASH_KEYWORDS = {"SHV", "SGOV", "BIL", "VMFXX", "SPAXX", "FDRXX"}
    _REIT_KEYWORDS = {"VNQ", "IYR", "SCHH", "REIT", "REM"}
    _INTL_KEYWORDS = {"VXUS", "VEA", "VWO", "EFA", "EEM", "IEFA", "IXUS"}
    _COMMODITY_KEYWORDS = {"GLD", "IAU", "SLV", "USO", "DJP", "PDBC"}

    def _classify_asset_type(self, symbol: str, sector: str) -> str:
        """Classify a ticker into a broad asset type."""
        sym = symbol.upper()
        if sym in self._BOND_KEYWORDS or "BOND" in sym:
            return "Bonds"
        if sym in self._CASH_KEYWORDS:
            return "Cash Equivalents"
        if sym in self._REIT_KEYWORDS or sector in ("Real Estate",):
            return "Real Estate"
        if sym in self._INTL_KEYWORDS:
            return "International Equities"
        if sym in self._COMMODITY_KEYWORDS:
            return "Commodities"
        return "Equities"

    def _resolve_holdings(self, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return holdings list from context, fetching live prices when only a
        {ticker: shares} portfolio dict is provided (from the UI session state).
        """
        if not context:
            return []
        if context.get("holdings"):
            return context["holdings"]

        portfolio_dict = context.get("portfolio", {})
        if not portfolio_dict:
            return []

        import yfinance as yf
        holdings: List[Dict[str, Any]] = []
        for symbol, shares in portfolio_dict.items():
            try:
                info = yf.Ticker(symbol).fast_info
                price = float(getattr(info, "last_price", 0) or 0)
                holdings.append({"symbol": symbol, "shares": float(shares), "price": price, "sector": "Unknown"})
            except Exception:
                holdings.append({"symbol": symbol, "shares": float(shares), "price": 0.0, "sector": "Unknown"})
        return holdings

    def _enrich_with_expense_ratios(self, holdings: List[Dict[str, Any]]) -> None:
        """
        Fetch expense ratio and sector from yfinance for each holding in-place.
        Uses fast_info.quote_type to skip the heavy .info call for plain equities,
        which have no expense ratio by definition.
        """
        import yfinance as yf
        for h in holdings:
            if "expense_ratio" in h and h.get("sector", "Unknown") != "Unknown":
                continue  # already enriched
            sym = h["symbol"]
            try:
                fi = yf.Ticker(sym).fast_info
                quote_type = (getattr(fi, "quote_type", None) or "EQUITY").upper()
                if quote_type in ("ETF", "MUTUALFUND"):
                    info = yf.Ticker(sym).info
                    h["expense_ratio"] = float(
                        info.get("annualReportExpenseRatio") or
                        info.get("netExpenseRatio") or 0.0
                    )
                    h.setdefault("sector", info.get("fundFamily") or "Unknown")
                else:
                    h["expense_ratio"] = 0.0
                    h.setdefault("sector", "Unknown")
                _tracer.detail("expense_ratio_fetched", symbol=sym,
                               expense_ratio=h["expense_ratio"], sector=h["sector"],
                               quote_type=quote_type)
            except Exception as e:
                h.setdefault("expense_ratio", 0.0)
                h.setdefault("sector", "Unknown")
                _tracer.warn("expense_ratio_fetch_failed", symbol=sym, error=str(e))

    def _has_portfolio_data(self, context: Optional[Dict[str, Any]]) -> bool:
        """Return True if real holdings are available (directly or via portfolio dict)."""
        return bool(self._resolve_holdings(context))

    def _answer_without_portfolio(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Answer a portfolio query when no holdings data is available."""
        rag_docs = (context or {}).get("documents", [])
        rag_snippet = ""
        if rag_docs:
            snippets = [d.get("content", d.get("page_content", "")) for d in rag_docs[:3] if isinstance(d, dict)]
            if snippets:
                rag_snippet = "\n\nRelevant background:\n" + "\n---\n".join(snippets)

        context_block = (
            f"{rag_snippet}\n\n"
            "No personal portfolio data has been provided. "
            "Answer with detailed general financial guidance, concepts, and best practices relevant to the question."
        )
        return self._call_llm(query, context_block)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze a user's portfolio.

        Args:
            query: Portfolio-related query
            context: Portfolio data, holdings, etc.

        Returns:
            Portfolio analysis with recommendations
        """
        _tracer.step("process_start", query_len=len(query),
                     context_keys=list((context or {}).keys()))
        self.add_to_history("user", query)

        if not self._has_portfolio_data(context):
            _tracer.decision("no_portfolio_data", path="answer_without_portfolio")
            response = self._answer_without_portfolio(query, context)
            self.add_to_history("assistant", response)
            return response

        _tracer.decision("has_portfolio_data", path="full_analysis_pipeline")

        t0 = time.perf_counter()
        portfolio_data = self._extract_portfolio_data(context)
        _tracer.timing("extract_portfolio_data", time.perf_counter() - t0,
                       holdings=len(portfolio_data["holdings"]),
                       total_value=round(portfolio_data["total_value"], 2))

        t0 = time.perf_counter()
        metrics = self._calculate_portfolio_metrics(portfolio_data)
        _tracer.timing("calculate_metrics", time.perf_counter() - t0,
                       diversification_score=metrics["diversification_score"],
                       hhi=metrics["hhi"],
                       risk_level=metrics["portfolio_risk_level"],
                       weighted_expense_ratio=metrics["weighted_expense_ratio"])

        visualizations = self._generate_visualizations(metrics, portfolio_data)
        recommendations = self._generate_recommendations(metrics, portfolio_data)
        _tracer.detail("recommendations_generated", count=len(recommendations))

        context_block = (
            "\n\nStructured portfolio analysis:\n"
            + json.dumps({
                "key_metrics": metrics["key_metrics"],
                "asset_allocation": metrics["asset_allocation"],
                "sector_allocation": metrics["sector_allocation"],
                "concentration_risk": metrics["concentration_risk"],
                "risk_level": metrics["portfolio_risk_level"],
                "investment_horizon": portfolio_data.get("investment_horizon", ""),
                "recommendations": recommendations,
                "position_chart": visualizations["concentration_chart"],
            }, default=str, indent=2)
        )
        response = self._call_llm(query, context_block)
        self.add_to_history("assistant", response)
        _tracer.step("process_complete", response_len=len(response))
        return response

    def _extract_portfolio_data(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract portfolio data from context."""
        portfolio_data = {
            "holdings": [],
            "total_value": 0.0,
            "cash": 0.0,
            "risk_profile": "moderate",
            "investment_horizon": "5-10 years"
        }

        ctx = context or {}
        portfolio_data["holdings"] = self._resolve_holdings(context)
        portfolio_data["cash"] = ctx.get("cash", 0.0)
        portfolio_data["risk_profile"] = ctx.get("risk_profile", "moderate")
        portfolio_data["investment_horizon"] = ctx.get("investment_horizon", "5-10 years")

        # Enrich holdings with expense ratios and sector info
        t0 = time.perf_counter()
        self._enrich_with_expense_ratios(portfolio_data["holdings"])
        _tracer.timing("enrich_expense_ratios", time.perf_counter() - t0,
                       holdings_enriched=len(portfolio_data["holdings"]))

        # Calculate total value
        holding_value = sum(h.get("shares", 0) * h.get("price", 0) for h in portfolio_data["holdings"])
        portfolio_data["total_value"] = holding_value + portfolio_data["cash"]

        # Annotate each holding: value, weight, asset_type
        for h in portfolio_data["holdings"]:
            h["value"] = h.get("shares", 0) * h.get("price", 0)
            h["weight"] = h["value"] / portfolio_data["total_value"] if portfolio_data["total_value"] > 0 else 0
            h["asset_type"] = self._classify_asset_type(h.get("symbol", ""), h.get("sector", "Unknown"))
            _tracer.detail("holding_classified", symbol=h["symbol"],
                           asset_type=h["asset_type"], sector=h.get("sector"),
                           weight_pct=round(h["weight"] * 100, 2))

        return portfolio_data

    def _calculate_portfolio_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate portfolio metrics per widely accepted frameworks.

        Diversification score: derived from the Herfindahl-Hirschman Index (HHI).
          HHI = sum(weight_i^2) scaled 0-1.  Score = (1 - HHI) * 100 (higher = more diversified).
          Reference: SEC/DOJ use HHI for concentration analysis.

        Risk level: based on equity-equivalent weight following Vanguard Target Retirement
          allocation methodology (>= 80% equities → Aggressive, etc.).
        """
        metrics: Dict[str, Any] = {
            "asset_allocation": {},
            "sector_allocation": {},
            "diversification_score": 0.0,
            "hhi": 0.0,
            "concentration_risk": 0.0,
            "portfolio_risk_level": "Moderate",
            "weighted_expense_ratio": 0.0,
            "key_metrics": {},
        }

        total_value = portfolio_data["total_value"]
        holdings = portfolio_data["holdings"]

        # ── Asset-type and sector buckets ────────────────────────────────────
        asset_buckets: Dict[str, float] = {}
        sector_allocation: Dict[str, float] = {}

        for h in holdings:
            val = h.get("value", 0.0)
            asset_type = h.get("asset_type", "Equities")
            asset_buckets[asset_type] = asset_buckets.get(asset_type, 0.0) + val
            sector = h.get("sector", "Unknown")
            sector_allocation[sector] = sector_allocation.get(sector, 0.0) + val

        if portfolio_data["cash"] > 0:
            asset_buckets["Cash Equivalents"] = asset_buckets.get("Cash Equivalents", 0.0) + portfolio_data["cash"]

        # Convert to percentages
        def to_pct(d: Dict[str, float]) -> Dict[str, float]:
            return {k: round(v / total_value * 100, 2) if total_value else 0.0 for k, v in d.items()}

        metrics["asset_allocation"] = to_pct(asset_buckets)
        metrics["sector_allocation"] = to_pct(sector_allocation)

        # ── HHI-based diversification score ──────────────────────────────────
        weights = [h.get("weight", 0.0) for h in holdings]
        hhi = sum(w ** 2 for w in weights)  # 1.0 = fully concentrated, 1/n = perfectly equal
        metrics["hhi"] = round(hhi, 4)
        metrics["diversification_score"] = round((1.0 - hhi) * 100, 1)

        # ── Concentration risk (largest single position) ──────────────────────
        metrics["concentration_risk"] = round(max((w * 100 for w in weights), default=0.0), 2)

        # ── Weighted average expense ratio (source: fund info from yfinance) ─
        if total_value > 0:
            metrics["weighted_expense_ratio"] = round(
                sum(h.get("expense_ratio", 0.0) * h.get("weight", 0.0) for h in holdings) * 100, 4
            )

        # ── Risk level (equity-equivalent weight, Vanguard methodology) ──────
        equity_pct = (
            metrics["asset_allocation"].get("Equities", 0.0) +
            metrics["asset_allocation"].get("International Equities", 0.0) +
            metrics["asset_allocation"].get("Real Estate", 0.0)
        )
        if equity_pct >= 80:
            risk_level = "Aggressive"
        elif equity_pct >= 60:
            risk_level = "Moderate-Aggressive"
        elif equity_pct >= 40:
            risk_level = "Moderate"
        elif equity_pct >= 20:
            risk_level = "Conservative"
        else:
            risk_level = "Very Conservative"
        metrics["portfolio_risk_level"] = risk_level

        # ── Key metrics summary ───────────────────────────────────────────────
        metrics["key_metrics"] = {
            "Total Portfolio Value": f"${total_value:,.2f}",
            "Total Holdings": len(holdings),
            "Asset Types": len(asset_buckets),
            "Sectors": len(sector_allocation),
            "Largest Position": f"{metrics['concentration_risk']:.1f}%",
            "HHI (concentration)": f"{metrics['hhi']:.4f}",
            "Diversification Score": f"{metrics['diversification_score']:.1f}/100",
            "Weighted Expense Ratio": f"{metrics['weighted_expense_ratio']:.4f}%",
            "Risk Level": risk_level,
        }

        return metrics

    def _generate_visualizations(self, metrics: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations (ASCII representations and descriptions)."""
        visualizations = {
            "allocation_bar_chart": "",
            "sector_distribution": "",
            "concentration_chart": "",
            "risk_gauge": ""
        }

        # Asset allocation bar chart
        allocation = metrics["asset_allocation"]
        visualizations["allocation_bar_chart"] = self._create_bar_chart(allocation, "Asset Allocation")

        # Sector distribution pie chart representation
        sectors = metrics["sector_allocation"]
        visualizations["sector_distribution"] = self._create_sector_chart(sectors)

        # Concentration chart
        holdings = sorted(portfolio_data["holdings"], key=lambda x: x.get("weight", 0), reverse=True)
        visualizations["concentration_chart"] = self._create_concentration_chart(holdings)

        # Risk gauge
        risk_level = metrics["portfolio_risk_level"]
        visualizations["risk_gauge"] = self._create_risk_gauge(risk_level)

        return visualizations

    def _create_bar_chart(self, data: Dict[str, float], title: str) -> str:
        """Create ASCII bar chart."""
        chart_lines = [f"\n{title}:\n"]
        max_value = max(data.values()) if data.values() else 1
        
        for label, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((value / max_value) * 30) if max_value > 0 else 0
            bar = "█" * bar_length
            chart_lines.append(f"{label:15} {bar:30} {value:6.2f}%\n")
        
        return "".join(chart_lines)

    def _create_sector_chart(self, sectors: Dict[str, float]) -> str:
        """Create sector distribution representation."""
        chart_lines = ["\nSector Distribution:\n"]
        
        for sector, percentage in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            # Create a simple pie representation
            slices = int(percentage / 5)  # Each slice represents ~5%
            chart_lines.append(f"{sector:20} {'◆' * slices:30} {percentage:6.2f}%\n")
        
        return "".join(chart_lines)

    def _create_concentration_chart(self, holdings: List[Dict[str, Any]]) -> str:
        """Create concentration risk chart."""
        chart_lines = ["\nTop 10 Holdings by Weight:\n"]
        
        for i, holding in enumerate(holdings[:10], 1):
            symbol = holding.get("symbol", "")
            weight = holding.get("weight", 0) * 100
            value = holding.get("value", 0)
            bar_length = int(weight / 2)  # Scale for readability
            bar = "▮" * bar_length
            chart_lines.append(f"{i:2}. {symbol:8} {bar:40} {weight:5.2f}% (${value:,.0f})\n")
        
        return "".join(chart_lines)

    def _create_risk_gauge(self, risk_level: str) -> str:
        """Create risk level gauge."""
        risk_levels = {
            "Very Conservative":   "▯▯▯▯▮ VERY LOW RISK",
            "Conservative":        "▯▯▯▮▮ LOW RISK",
            "Moderate":            "▯▯▮▮▮ MODERATE RISK",
            "Moderate-Aggressive": "▯▮▮▮▮ MODERATE-HIGH RISK",
            "Aggressive":          "▮▮▮▮▮ HIGH RISK",
        }
        
        gauge = risk_levels.get(risk_level, "▯▮▮▮▯ MODERATE RISK")
        return f"\nRisk Profile: {gauge}\n"

    def _generate_recommendations(self, metrics: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[str]:
        """Provide recommendations based on risk profile."""
        recommendations = []
        
        # Diversification recommendations
        div_score = metrics["diversification_score"]
        if div_score < 50:
            recommendations.append("🔴 Diversification is LOW. Consider adding positions in underrepresented sectors.")
        elif div_score < 70:
            recommendations.append("🟡 Diversification could be improved. Look into adding ETFs or funds for broader exposure.")
        else:
            recommendations.append("🟢 Portfolio is well-diversified across holdings and sectors.")

        # Concentration risk recommendations
        conc_risk = metrics["concentration_risk"]
        if conc_risk > 20:
            recommendations.append(f"🔴 Concentration Risk is HIGH (largest position: {conc_risk}%). Consider trimming the largest position.")
        elif conc_risk > 15:
            recommendations.append(f"🟡 Concentration Risk is moderate ({conc_risk}%). Monitor the largest position for rebalancing opportunities.")
        else:
            recommendations.append(f"🟢 Concentration Risk is well-controlled ({conc_risk}%).")

        # Risk level alignment
        risk_profile = portfolio_data.get("risk_profile", "moderate")
        portfolio_risk = metrics["portfolio_risk_level"]
        
        risk_mapping = {"conservative": "Conservative", "moderate": "Moderate", "aggressive": "High"}
        target_risk = risk_mapping.get(risk_profile.lower(), "Moderate")
        
        if portfolio_risk != target_risk:
            recommendations.append(f"🟡 Risk alignment: Your portfolio is '{portfolio_risk}' but your profile is '{target_risk}'. Consider rebalancing.")
        else:
            recommendations.append(f"🟢 Portfolio risk level aligns with your '{portfolio_risk}' risk profile.")

        # Asset allocation recommendations
        allocation = metrics["asset_allocation"]
        equity_pct = allocation.get("Equities", 0) + allocation.get("International Equities", 0)
        cash_pct = allocation.get("Cash Equivalents", 0)

        if cash_pct > 20:
            recommendations.append(f"💡 You have {cash_pct:.1f}% in cash/equivalents. Consider deploying excess cash into underweight asset types.")

        if equity_pct < 40 and risk_profile.lower() == "aggressive":
            recommendations.append("💡 For an aggressive profile, increase equity exposure by reducing bonds/cash (target ~90% equities per Vanguard methodology).")

        # Expense ratio commentary
        wer = metrics.get("weighted_expense_ratio", 0.0)
        if wer > 0.5:
            recommendations.append(f"💰 Weighted expense ratio is {wer:.2f}% — above the low-cost threshold. Consider switching high-fee funds to index equivalents (source: Morningstar 2024 fee study: avg index fund 0.05%).")
        elif wer > 0 and wer <= 0.2:
            recommendations.append(f"💰 Weighted expense ratio is {wer:.4f}% — competitive with low-cost index funds (Morningstar benchmark: < 0.20%).")

        # Investment horizon recommendations
        horizon = portfolio_data.get("investment_horizon", "")
        if "1-3" in horizon:
            recommendations.append("⏱️  With a 1-3 year horizon, consider moving to more conservative assets.")
        elif "20+" in horizon:
            recommendations.append("⏱️  With a 20+ year horizon, you can afford to take more equity risk for growth.")

        # Sector-specific recommendations
        sectors = metrics["sector_allocation"]
        if "Technology" in sectors and sectors["Technology"] > 40:
            recommendations.append("🔴 Tech exposure is high (>40%). Consider diversifying into other sectors.")
        
        has_equity = any(
            h.get("asset_type", "") in ("Equities", "International Equities")
            for h in portfolio_data["holdings"]
        )
        if has_equity and ("Healthcare" not in sectors or sectors.get("Healthcare", 0) < 5):
            recommendations.append("💡 Healthcare sector is underrepresented. It provides defensive characteristics.")

        return recommendations

