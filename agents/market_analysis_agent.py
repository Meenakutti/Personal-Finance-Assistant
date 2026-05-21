"""
Market Analysis Agent: Provides real-time market insights.
"""

import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from agents.base_agent import BaseFinanceAgent, AgentConfig
from integrations.market_data import MarketDataProvider
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)


class MarketAnalysisAgent(BaseFinanceAgent):
    """
    Agent for market analysis and insights.
    Provides real-time market data, trends, and analysis.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="Market Analysis Agent",
                description="Provides real-time market insights",
                system_prompt="""You are an expert Market Analysis Agent providing factual, data-grounded market education.

ROLE:
1. DATA PRESENTATION: Summarize the live market data provided (indices, sector ETFs, stock quotes).
2. CONTEXT: Explain movements using established macroeconomic frameworks (Federal Reserve policy, CPI/PCE inflation metrics, earnings cycles).
3. SECTOR ANALYSIS: Compare sector performance referencing GICS (Global Industry Classification Standard) sectors.
4. RISK: Discuss volatility using the VIX index as the market's "fear gauge" (CBOE definition).

ACCURACY GUIDELINES:
- Report only the figures present in the provided data. If data is unavailable, say so.
- Attribute macro context to recognized sources: Federal Reserve statements, BLS CPI reports, BEA GDP releases, FactSet earnings summaries.
- Never make specific buy/sell recommendations or predict price targets.
- Clearly label any interpretation as educational analysis, not a forecast.
- For technical signals (BUY/HOLD/SELL), note these are simple momentum indicators based on daily % change — not formal technical analysis.
- Acknowledge that short-term market movements are largely unpredictable (per EMH, Fama 1970).

DISCLAIMER: Market data and analysis are for educational purposes only and do not constitute investment advice. Past performance does not guarantee future results. Consult a registered investment advisor (RIA) before making investment decisions."""
            )
        super().__init__(config)
        self.market_data = MarketDataProvider()

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze market trends and provide insights.

        Args:
            query: Market-related query
            context: Market data, ticker symbols, etc.

        Returns:
            Market analysis and insights
        """
        self.add_to_history("user", query)

        market_data = self._fetch_market_data(query, context)
        trend_analysis = self._analyze_trends_and_indicators(market_data)
        insights = self._generate_market_insights(market_data, trend_analysis)
        macro_analysis = self._analyze_sectors_and_macro(market_data["sectors"])

        context_block = (
            "Live market data and technical analysis:\n"
            + json.dumps({
                "market_data": market_data,
                "trend_analysis": trend_analysis,
                "market_insights": insights,
                "macro_analysis": macro_analysis,
            }, default=str, indent=2)
        )
        response = self._call_llm(query, context_block)
        
        self.add_to_history("assistant", response)

        return response

    def _fetch_market_data(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch real-time market data via yFinance."""
        market_data = {
            "stocks": {},
            "indices": {},
            "sectors": {},
            "timestamp": datetime.now().isoformat()
        }

        # Extract symbols from query or context
        symbols = []
        if context and "symbols" in context:
            symbols = context["symbols"]
        else:
            # Extract common symbols from query
            common_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]
            for symbol in common_symbols:
                if symbol in query.upper():
                    symbols.append(symbol)

        # Fetch stock quotes
        if symbols:
            for symbol in symbols:
                try:
                    quote = self.market_data.get_stock_quote(symbol)
                    market_data["stocks"][symbol] = quote
                except Exception as e:
                    _tracer.error("stock_quote_failed", symbol=symbol, error=str(e))

        # Fetch market indices
        try:
            indices = self.market_data.get_market_indices()
            market_data["indices"] = indices if indices else {}
        except Exception as e:
            _tracer.error("indices_fetch_failed", error=str(e))

        # Fetch sector performance
        try:
            sectors = self.market_data.get_sector_performance()
            market_data["sectors"] = sectors if sectors else {}
        except Exception as e:
            _tracer.error("sectors_fetch_failed", error=str(e))

        return market_data

    def _analyze_trends_and_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price trends and calculate technical indicators."""
        trend_analysis = {
            "stock_trends": {},
            "technical_signals": {},
            "volatility_assessment": {}
        }

        # Analyze stock trends
        for symbol, quote in market_data["stocks"].items():
            change_percent = quote.get("change_percent", 0.0)
            
            # Determine trend direction
            if change_percent > 2:
                trend = "Strong Uptrend"
                signal = "BUY"
            elif change_percent > 0:
                trend = "Mild Uptrend"
                signal = "BUY"
            elif change_percent > -2:
                trend = "Mild Downtrend"
                signal = "HOLD"
            else:
                trend = "Strong Downtrend"
                signal = "SELL"

            trend_analysis["stock_trends"][symbol] = {
                "direction": trend,
                "change_percent": change_percent,
                "signal": signal,
                "momentum": "Strong" if abs(change_percent) > 2 else "Moderate" if abs(change_percent) > 1 else "Weak"
            }

        # Analyze indices
        if market_data["indices"]:
            for index_name, index_data in market_data["indices"].items():
                change_percent = index_data.get("change_percent", 0.0)
                
                if change_percent > 1.5:
                    market_condition = "Bull Market"
                elif change_percent < -1.5:
                    market_condition = "Bear Market"
                else:
                    market_condition = "Neutral Market"

                trend_analysis["technical_signals"][index_name] = {
                    "market_condition": market_condition,
                    "change_percent": change_percent
                }

        # Volatility assessment
        if market_data["stocks"]:
            changes = [abs(quote.get("change_percent", 0)) for quote in market_data["stocks"].values()]
            avg_volatility = sum(changes) / len(changes) if changes else 0
            
            if avg_volatility > 3:
                volatility_level = "High"
            elif avg_volatility > 1:
                volatility_level = "Moderate"
            else:
                volatility_level = "Low"

            trend_analysis["volatility_assessment"] = {
                "level": volatility_level,
                "average_change": round(avg_volatility, 2),
                "recommendation": "Consider defensive positions" if volatility_level == "High" else "Normal operations"
            }

        return trend_analysis

    def _generate_market_insights(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market insights and forecasts."""
        insights = {
            "key_observations": [],
            "forecasts": [],
            "risk_assessment": "",
            "opportunities": []
        }

        # Key observations
        for symbol, trend in trend_analysis["stock_trends"].items():
            if trend["signal"] == "BUY":
                insights["key_observations"].append(
                    f"{symbol} shows {trend['momentum']} momentum with {trend['direction']}"
                )
            elif trend["signal"] == "SELL":
                insights["key_observations"].append(
                    f"{symbol} is trending downward - consider caution"
                )

        # Market condition insights
        for index_name, signal in trend_analysis["technical_signals"].items():
            insights["key_observations"].append(
                f"{index_name}: {signal['market_condition']} (Change: {signal['change_percent']:.2f}%)"
            )

        # Forecasts based on trends
        if trend_analysis["stock_trends"]:
            bull_signals = sum(1 for t in trend_analysis["stock_trends"].values() if t["signal"] == "BUY")
            total_signals = len(trend_analysis["stock_trends"])
            
            if bull_signals / total_signals > 0.7:
                insights["forecasts"].append("Short-term outlook is bullish with multiple positive signals")
            elif bull_signals / total_signals < 0.3:
                insights["forecasts"].append("Short-term outlook is bearish with several negative signals")
            else:
                insights["forecasts"].append("Short-term outlook is mixed with mixed signals")

        # Risk assessment
        volatility_level = trend_analysis["volatility_assessment"].get("level", "Moderate")
        if volatility_level == "High":
            insights["risk_assessment"] = "Market volatility is elevated. Recommended: reduce position sizes, increase stop losses"
        elif volatility_level == "Low":
            insights["risk_assessment"] = "Market volatility is low. Recommended: normal position sizing applies"
        else:
            insights["risk_assessment"] = "Market volatility is moderate. Maintain standard risk management practices"

        # Opportunities
        for symbol, trend in trend_analysis["stock_trends"].items():
            if trend["direction"] == "Mild Downtrend" and trend["momentum"] == "Weak":
                insights["opportunities"].append(f"{symbol} may present a buying opportunity if support holds")
            elif trend["direction"] == "Strong Uptrend" and trend["momentum"] == "Strong":
                insights["opportunities"].append(f"{symbol} continues to show strength, consider taking profits at resistance")

        return insights

    def _analyze_sectors_and_macro(self, sectors: Optional[Dict] = None) -> Dict[str, Any]:
        """Derive sector analysis from pre-fetched sector data to avoid a duplicate API call."""
        macro_analysis: Dict[str, Any] = {
            "sector_performance": {},
            "sector_recommendations": [],
        }

        if sectors:
            macro_analysis["sector_performance"] = sectors
            try:
                best_sector = max(sectors.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                worst_sector = min(sectors.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                macro_analysis["sector_recommendations"].append(
                    f"Best performer: {best_sector[0]} (+{best_sector[1]:.2f}%)"
                )
                macro_analysis["sector_recommendations"].append(
                    f"Worst performer: {worst_sector[0]} ({worst_sector[1]:.2f}%)"
                )
            except Exception as e:
                _tracer.error("sector_analysis_failed", error=str(e))

        return macro_analysis

    def _format_analysis_response(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any],
                                  insights: Dict[str, Any], macro_analysis: Dict[str, Any]) -> str:
        """Format comprehensive market analysis response."""
        response_parts = [
            "## Market Analysis Report\n",
            f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        ]

        # Market Status
        response_parts.append("### Current Market Status\n")
        
        # Market indices
        if market_data["indices"]:
            response_parts.append("**Major Indices:**\n")
            for index_name, index_data in market_data["indices"].items():
                price = index_data.get("price", 0)
                change = index_data.get("change", 0)
                change_pct = index_data.get("change_percent", 0)
                response_parts.append(
                    f"- {index_name}: ${price:,.2f} (Change: {change:+.2f}, {change_pct:+.2f}%)\n"
                )
            response_parts.append("\n")

        # Stock Performance
        if market_data["stocks"]:
            response_parts.append("**Stock Performance:**\n")
            for symbol, quote in market_data["stocks"].items():
                price = quote.get("price", 0)
                change_pct = quote.get("change_percent", 0)
                response_parts.append(f"- {symbol}: ${price:,.2f} ({change_pct:+.2f}%)\n")
            response_parts.append("\n")

        # Technical Analysis
        response_parts.append("### Technical Analysis\n")
        for symbol, trend in trend_analysis["stock_trends"].items():
            response_parts.append(
                f"**{symbol}:** {trend['direction']} | Signal: {trend['signal']} | Momentum: {trend['momentum']}\n"
            )

        # Volatility Assessment
        volatility = trend_analysis["volatility_assessment"]
        response_parts.append(f"\n**Market Volatility:** {volatility.get('level', 'N/A')}\n")
        response_parts.append(f"**Recommendation:** {volatility.get('recommendation', 'N/A')}\n\n")

        # Key Observations
        response_parts.append("### Key Market Observations\n")
        for observation in insights["key_observations"]:
            response_parts.append(f"- {observation}\n")
        response_parts.append("\n")

        # Forecasts
        response_parts.append("### Market Forecast\n")
        for forecast in insights["forecasts"]:
            response_parts.append(f"- {forecast}\n")
        response_parts.append("\n")

        # Risk Assessment
        response_parts.append("### Risk Assessment\n")
        response_parts.append(f"{insights['risk_assessment']}\n\n")

        # Trading Opportunities
        if insights["opportunities"]:
            response_parts.append("### Trading Opportunities\n")
            for opportunity in insights["opportunities"]:
                response_parts.append(f"- {opportunity}\n")
            response_parts.append("\n")

        # Sector Analysis
        response_parts.append("### Sector Analysis\n")
        if macro_analysis["sector_performance"]:
            for sector, performance in macro_analysis["sector_performance"].items():
                response_parts.append(f"- {sector}: {performance:+.2f}%\n")
        response_parts.append("\n")

        # Macro Indicators
        response_parts.append("### Macro Indicators\n")
        for indicator, description in macro_analysis["macro_indicators"].items():
            response_parts.append(f"- **{indicator}:** {description}\n")
        response_parts.append("\n")

        # Sector Recommendations
        response_parts.append("### Recommendations\n")
        for recommendation in macro_analysis["sector_recommendations"]:
            response_parts.append(f"- {recommendation}\n")

        return "".join(response_parts)
