"""
News Synthesizer Agent: Summarizes and contextualizes financial news.
"""

import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from agents.base_agent import BaseFinanceAgent, AgentConfig
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)


class NewsSynthesizerAgent(BaseFinanceAgent):
    """
    Agent for synthesizing and contextualizing financial news.
    Aggregates news, identifies relevance, and provides context.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="News Synthesizer Agent",
                description="Summarizes and contextualizes financial news",
                system_prompt="""You are an expert News Synthesizer Agent providing factual, balanced financial news summaries.

ROLE:
1. SUMMARIZE: Provide clear, neutral summaries of the provided news headlines and data.
2. CONTEXTUALIZE: Explain why a story matters using established financial principles.
3. BALANCE: Present multiple perspectives; avoid sensationalism or speculation.
4. CONNECT: Link news to relevant economic indicators (CPI, Fed rate decisions, earnings) and academic frameworks where applicable.

ACCURACY GUIDELINES:
- Only report on the specific news items provided in the context data. Do not fabricate headlines or events.
- Cite the source publisher for each story (e.g., "per Reuters", "per Bloomberg", "per Yahoo Finance").
- Distinguish clearly between: (a) reported facts from the news, (b) established economic context, and (c) your educational interpretation.
- Avoid predicting market movements or suggesting specific trades based on news.
- Reference widely accepted market theory where relevant (e.g., "efficient markets typically price in earnings surprises quickly — per the EMH, Fama 1970").

When a story is unclear or the source is unknown, say so explicitly rather than guessing.

Use clear headlines and structured formats.
Always distinguish between facts and speculation or analyst opinions.

DISCLAIMER: News summaries are for informational purposes only and do not constitute investment advice. Always verify information from primary sources before making investment decisions."""
            )
        super().__init__(config)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Synthesize financial news and provide context.

        Args:
            query: News-related query or topic
            context: User portfolio, interests, etc.

        Returns:
            Synthesized news summary with relevance to user
        """
        self.add_to_history("user", query)

        news_articles = self._fetch_financial_news(query)
        ranked_news = self._filter_and_rank_by_relevance(news_articles, query, context)
        summaries = self._summarize_key_points(ranked_news[:8])
        portfolio_connections = self._connect_to_portfolio(ranked_news[:8], context)

        context_block = (
            "\n\nLive financial news articles (ranked by relevance):\n"
            + json.dumps(ranked_news[:8], default=str, indent=2)
            + "\n\nPre-computed key points:\n"
            + json.dumps(summaries, default=str, indent=2)
            + "\n\nPortfolio connections:\n"
            + json.dumps(portfolio_connections, default=str, indent=2)
        )
        response = self._call_llm(query, context_block)
        
        self.add_to_history("assistant", response)

        return response

    def _fetch_financial_news(self, query: str) -> List[Dict[str, Any]]:
        """Fetch live financial news via yfinance."""
        import yfinance as yf

        common_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]
        symbols = [s for s in common_symbols if s in query.upper()]
        if not symbols:
            symbols = ["SPY", "QQQ"]

        articles: List[Dict[str, Any]] = []
        seen: set = set()

        for symbol in symbols[:3]:
            try:
                news = yf.Ticker(symbol).news or []
                for item in news[:5]:
                    title = item.get("title", "")
                    if not title or title in seen:
                        continue
                    seen.add(title)
                    ts = item.get("providerPublishTime", 0)
                    t = title.lower()
                    if any(w in t for w in ("surge", "soar", "rally", "beat", "rise", "gain", "record", "high", "jump", "boost")):
                        impact = "positive"
                    elif any(w in t for w in ("fall", "drop", "plunge", "miss", "decline", "loss", "cut", "warn", "crash", "risk", "fear", "tumble")):
                        impact = "negative"
                    else:
                        impact = "neutral"
                    articles.append({
                        "title": title,
                        "source": item.get("publisher", "Unknown"),
                        "date": datetime.fromtimestamp(ts).isoformat() if ts else datetime.now().isoformat(),
                        "url": item.get("link", ""),
                        "symbols": item.get("relatedTickers", [symbol]),
                        "impact": impact,
                    })
            except Exception as e:
                _tracer.warn("yfinance_news_failed", symbol=symbol, error=str(e))

        return articles[:10]

    def _filter_and_rank_by_relevance(self, articles: List[Dict[str, Any]], query: str, 
                                      context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Filter and rank news articles by relevance to user."""
        ranked_articles = []
        
        # Get user portfolio symbols if available
        # UI passes {ticker: shares} dict; handle both shapes
        user_symbols = []
        if context and "portfolio" in context:
            portfolio = context["portfolio"]
            if isinstance(portfolio, dict):
                user_symbols = portfolio.get("symbols") or list(portfolio.keys())
        
        # Get user interests if available
        user_interests = []
        if context and "interests" in context:
            user_interests = context["interests"]
        
        # Score and rank each article
        for article in articles:
            relevance_score = 0
            match_reasons = []
            
            # Score based on portfolio symbols
            article_symbols = article.get("symbols", [])
            symbol_matches = len([s for s in article_symbols if s in user_symbols])
            if symbol_matches > 0:
                relevance_score += symbol_matches * 30
                match_reasons.append(f"Mentions {symbol_matches} portfolio holdings")
            
            # Score based on user interests
            article_title = article.get("title", "").lower()
            article_content = article.get("content", "").lower()
            for interest in user_interests:
                if interest.lower() in article_title or interest.lower() in article_content:
                    relevance_score += 20
                    match_reasons.append(f"Covers {interest}")
            
            # Score based on impact
            if article.get("impact") == "positive":
                relevance_score += 10
            elif article.get("impact") == "negative":
                relevance_score += 15  # Negative news often more important
            
            # Score based on recency
            article_date = datetime.fromisoformat(article.get("date", datetime.now().isoformat()))
            days_old = (datetime.now() - article_date).days
            recency_score = max(0, 20 - days_old * 2)
            relevance_score += recency_score
            
            article["relevance_score"] = relevance_score
            article["match_reasons"] = match_reasons
            ranked_articles.append(article)
        
        # Sort by relevance score (descending)
        ranked_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return ranked_articles

    def _summarize_key_points(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize key points and implications from news."""
        summaries = {
            "trending_topics": [],
            "market_implications": [],
            "risk_alerts": [],
            "opportunities": []
        }
        
        for article in articles:
            content = article.get("content", "")
            title = article.get("title", "")
            impact = article.get("impact", "neutral")
            symbols = article.get("symbols", [])
            
            # Extract trending topics
            if "earnings" in title.lower():
                summaries["trending_topics"].append(f"Corporate earnings beat expectations ({', '.join(symbols)})")
            elif "rate" in title.lower():
                summaries["trending_topics"].append("Interest rate changes affecting market")
            elif "supply chain" in content.lower():
                summaries["trending_topics"].append("Supply chain developments impacting industries")
            
            # Extract market implications
            if impact == "positive":
                summaries["market_implications"].append(
                    f"Positive: {title} - May support portfolio growth"
                )
            elif impact == "negative":
                summaries["market_implications"].append(
                    f"Negative: {title} - May impact portfolio values"
                )
            
            # Extract risk alerts
            if "shortage" in content.lower() or "decline" in content.lower():
                summaries["risk_alerts"].append(f"Watch: {title}")
            
            # Extract opportunities
            if "eases" in content.lower() or "growth" in content.lower():
                summaries["opportunities"].append(f"Potential opportunity: {title}")
        
        # Remove duplicates
        for key in summaries:
            summaries[key] = list(dict.fromkeys(summaries[key]))
        
        return summaries

    def _connect_to_portfolio(self, articles: List[Dict[str, Any]], 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Connect news to user's portfolio and interests."""
        portfolio_connections = {
            "affected_holdings": [],
            "sector_impact": {},
            "action_items": []
        }
        
        if not context or "portfolio" not in context:
            return portfolio_connections

        user_portfolio = context.get("portfolio", {})
        if isinstance(user_portfolio, dict):
            user_symbols = user_portfolio.get("symbols") or list(user_portfolio.keys())
        else:
            user_symbols = []
        
        # Find affected holdings
        for article in articles:
            article_symbols = article.get("symbols", [])
            affected = [s for s in article_symbols if s in user_symbols]
            
            if affected:
                portfolio_connections["affected_holdings"].append({
                    "symbols": affected,
                    "news": article.get("title", ""),
                    "action": "Review" if article.get("impact") == "negative" else "Monitor"
                })
        
        # Sector impact analysis
        sector_mapping = {
            "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
            "finance": ["JPM", "BLK", "GS"],
            "energy": ["XOM", "CVX", "COP"],
            "healthcare": ["JNJ", "UNH", "PFE"]
        }
        
        for sector, symbols in sector_mapping.items():
            sector_article_count = 0
            for article in articles:
                if any(s in article.get("symbols", []) for s in symbols):
                    sector_article_count += 1
            
            if sector_article_count > 0:
                portfolio_connections["sector_impact"][sector] = {
                    "mentions": sector_article_count,
                    "sentiment": "mixed" if sector_article_count > 2 else "neutral"
                }
        
        # Generate action items
        if portfolio_connections["affected_holdings"]:
            for holding in portfolio_connections["affected_holdings"]:
                if holding["action"] == "Review":
                    portfolio_connections["action_items"].append(
                        f"Review position in {', '.join(holding['symbols'])} - Negative news"
                    )
                else:
                    portfolio_connections["action_items"].append(
                        f"Monitor {', '.join(holding['symbols'])} for developments"
                    )
        
        # Add general rebalancing recommendation
        if len(portfolio_connections["affected_holdings"]) > 2:
            portfolio_connections["action_items"].append(
                "Consider portfolio rebalancing given multiple news items"
            )
        
        return portfolio_connections

    def _format_news_response(self, ranked_articles: List[Dict[str, Any]], 
                             summaries: Dict[str, Any], 
                             portfolio_connections: Dict[str, Any]) -> str:
        """Format comprehensive news synthesizer response."""
        response_parts = [
            "## Financial News Synthesis Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Articles Analyzed:** {len(ranked_articles)}\n\n"
        ]
        
        # Top Headlines
        response_parts.append("### Top Headlines\n")
        for i, article in enumerate(ranked_articles[:5], 1):
            title = article.get("title", "")
            source = article.get("source", "Unknown")
            symbols = article.get("symbols", [])
            score = article.get("relevance_score", 0)
            reasons = article.get("match_reasons", [])
            
            response_parts.append(f"**{i}. {title}**\n")
            response_parts.append(f"   Source: {source}\n")
            if symbols:
                response_parts.append(f"   Symbols: {', '.join(symbols)}\n")
            response_parts.append(f"   Relevance Score: {score}\n")
            if reasons:
                response_parts.append(f"   Why Relevant: {'; '.join(reasons)}\n")
            response_parts.append("\n")
        
        # Trending Topics
        if summaries["trending_topics"]:
            response_parts.append("### Trending Topics\n")
            for topic in summaries["trending_topics"]:
                response_parts.append(f"- {topic}\n")
            response_parts.append("\n")
        
        # Market Implications
        if summaries["market_implications"]:
            response_parts.append("### Market Implications\n")
            for implication in summaries["market_implications"]:
                response_parts.append(f"- {implication}\n")
            response_parts.append("\n")
        
        # Risk Alerts
        if summaries["risk_alerts"]:
            response_parts.append("### Risk Alerts\n")
            for alert in summaries["risk_alerts"]:
                response_parts.append(f"- {alert}\n")
            response_parts.append("\n")
        
        # Opportunities
        if summaries["opportunities"]:
            response_parts.append("### Opportunities\n")
            for opportunity in summaries["opportunities"]:
                response_parts.append(f"- {opportunity}\n")
            response_parts.append("\n")
        
        # Portfolio Impact
        if portfolio_connections["affected_holdings"]:
            response_parts.append("### Impact on Your Portfolio\n")
            response_parts.append("**Affected Holdings:**\n")
            for holding in portfolio_connections["affected_holdings"]:
                response_parts.append(f"- {', '.join(holding['symbols'])}: {holding['news']}\n")
                response_parts.append(f"  Action: {holding['action']}\n")
            response_parts.append("\n")
        
        # Sector Analysis
        if portfolio_connections["sector_impact"]:
            response_parts.append("### Sector Analysis\n")
            for sector, impact in portfolio_connections["sector_impact"].items():
                response_parts.append(
                    f"- **{sector.upper()}:** {impact['mentions']} news items | "
                    f"Sentiment: {impact['sentiment']}\n"
                )
            response_parts.append("\n")
        
        # Action Items
        if portfolio_connections["action_items"]:
            response_parts.append("### Recommended Actions\n")
            for action in portfolio_connections["action_items"]:
                response_parts.append(f"- {action}\n")
        
        return "".join(response_parts)
