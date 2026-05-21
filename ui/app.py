"""
Streamlit UI for the Personal Finance Assistant.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from typing import Optional, Dict, Any
from config.workflow import FinanceAssistantWorkflow
from integrations.market_data import MarketDataProvider


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "workflow" not in st.session_state:
        st.session_state.workflow = FinanceAssistantWorkflow()

    if "market_data" not in st.session_state:
        st.session_state.market_data = MarketDataProvider()

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}

    if "suggested_prompt" not in st.session_state:
        st.session_state.suggested_prompt = None


def render_header():
    """Render the application header."""
    st.set_page_config(
        page_title="Personal Finance Assistant",
        page_icon="💰",
        layout="wide"
    )

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("💰 Personal Finance Assistant")
        st.write("Your intelligent guide to financial literacy and wealth management")
    with col2:
        st.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*")


def render_sidebar_settings():
    """Render settings and suggested prompts in the sidebar."""
    with st.sidebar:
        st.header("⚙️ Settings")

        st.divider()

        temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Response Length", 100, 2000, 1000)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()
        with col2:
            if st.button("💾 Clear Cache", use_container_width=True):
                st.session_state.market_data.clear_cache()
                st.success("Cache cleared!")

        st.divider()
        st.subheader("💡 Suggested Questions")
        for suggestion in QA_SUGGESTED_PROMPTS:
            if st.button(suggestion, use_container_width=True, key=f"suggest_{suggestion[:20]}"):
                st.session_state.suggested_prompt = suggestion
                st.rerun()

        st.divider()
        st.caption("Personal Finance Assistant v1.0")


QA_SUGGESTED_PROMPTS = [
    "What is the right asset allocation strategy based on my age?",
    "How do I diversify my investment portfolio effectively?",
    "What are common behavioral mistakes investors make?",
    "Why is market timing risky and should I try to time the market?",
    "Should I add commodities like gold to my portfolio for inflation protection?",
]


def render_chat_page():
    """Render the main chat interface."""
    st.header("Chat with Your Finance Assistant")

    # 1. Prior Q&A at top — rendered from session state before anything else
    history = st.session_state.conversation_history
    if history:
        last_user = next((m for m in reversed(history) if m["role"] == "user"), None)
        last_asst = next((m for m in reversed(history) if m["role"] == "assistant"), None)
        if last_user:
            with st.chat_message("user"):
                st.write(last_user["content"])
        if last_asst:
            with st.chat_message("assistant"):
                st.write(last_asst["content"])
                if last_asst.get("agent"):
                    info = st.session_state.workflow.registry.get_agent_info(last_asst["agent"])
                    label = info["name"] if info else last_asst["agent"].replace("_", " ").title()
                    st.caption(f"Handled by: {label}")
    else:
        st.info("Ask a question or pick a suggestion from the sidebar to get started.")

    # 2. Container for new Q&A — positioned directly below prior Q&A
    new_qa = st.container()

    # 3. Chat input — Streamlit always pins this to viewport bottom
    typed = st.chat_input("Ask me about finance, investments, taxes, or your financial goals...")

    # 4. Resolve prompt source
    prompt = None
    if st.session_state.get("suggested_prompt"):
        prompt = st.session_state.suggested_prompt
        st.session_state.suggested_prompt = None
    elif typed:
        prompt = typed

    # 5. Process and render new Q&A inside the container (below prior Q&A)
    if prompt:
        prior_history = list(st.session_state.conversation_history)

        with new_qa:
            with st.chat_message("user"):
                st.write(prompt)

            with st.spinner("Thinking..."):
                try:
                    output = st.session_state.workflow.process_query(
                        prompt,
                        context={"portfolio": st.session_state.portfolio},
                        conversation_history=prior_history,
                    )
                    response = output.get("final_response", "I couldn't process your query.")
                    routed_agents = output.get("routed_agents", [])
                except Exception as e:
                    response = f"Error processing query: {str(e)}"
                    routed_agents = []

            with st.chat_message("assistant"):
                st.write(response)
                active_agent = routed_agents[0] if routed_agents else None
                if active_agent:
                    info = st.session_state.workflow.registry.get_agent_info(active_agent)
                    label = info["name"] if info else active_agent.replace("_", " ").title()
                    st.caption(f"Handled by: {label}")

        # Persist to session state so next render shows it at top
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        active_agent = routed_agents[0] if routed_agents else None
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response,
            "agent": active_agent,
        })


def _fetch_portfolio_holdings(portfolio_dict: Dict[str, float]) -> list:
    """Fetch live prices and build holdings list from {ticker: shares} dict."""
    holdings = []
    for symbol, shares in portfolio_dict.items():
        try:
            info = yf.Ticker(symbol).fast_info
            price = float(getattr(info, "last_price", 0) or 0)
        except Exception:
            price = 0.0
        holdings.append({"symbol": symbol, "shares": shares, "price": price,
                         "value": shares * price})
    return holdings


def render_portfolio_page():
    """Render portfolio analysis page."""
    st.header("Portfolio Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add Stock")
        ticker = st.text_input("Ticker Symbol", placeholder="AAPL").upper().strip()
        shares = st.number_input("Number of Shares", min_value=0.0, step=1.0)
        add_col, remove_col = st.columns(2)
        with add_col:
            if st.button("Add", use_container_width=True):
                if not ticker:
                    st.warning("Please enter a ticker symbol (e.g. AAPL, TSLA, VTI) before adding shares.")
                elif shares <= 0:
                    st.warning("Please enter a number of shares greater than 0.")
                else:
                    existing = st.session_state.portfolio.get(ticker, 0.0)
                    st.session_state.portfolio[ticker] = existing + shares
                    st.success(f"Added {shares} shares of {ticker} (total: {existing + shares})")
                    st.rerun()
        with remove_col:
            if st.button("Remove", use_container_width=True):
                if ticker and ticker in st.session_state.portfolio:
                    del st.session_state.portfolio[ticker]
                    st.rerun()

    with col2:
        st.subheader("Current Holdings")
        if st.session_state.portfolio:
            for sym, sh in st.session_state.portfolio.items():
                st.write(f"- **{sym}**: {sh} shares")
        else:
            st.info("No holdings yet. Add stocks to get started!")

    if st.session_state.portfolio:
        st.divider()
        if st.button("Analyze Portfolio", type="primary"):
            with st.spinner("Fetching live prices and analyzing..."):
                holdings = _fetch_portfolio_holdings(st.session_state.portfolio)
                output = st.session_state.workflow.process_query(
                    "Analyze my portfolio: diversification, risk, and recommendations.",
                    context={"holdings": holdings},
                )
                st.session_state["portfolio_analysis"] = output.get("final_response", "")
                st.session_state["portfolio_holdings_data"] = holdings

        if st.session_state.get("portfolio_analysis"):
            st.markdown(st.session_state["portfolio_analysis"])

        holdings_data = st.session_state.get("portfolio_holdings_data") or \
                        _fetch_portfolio_holdings(st.session_state.portfolio)
        holdings_with_value = [h for h in holdings_data if h.get("value", 0) > 0]

        if holdings_with_value:
            st.divider()
            st.subheader("Portfolio Visualizations")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_pie = px.pie(
                    holdings_with_value,
                    names="symbol",
                    values="value",
                    title="Holdings by Value",
                    hole=0.35,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                fig_bar = px.bar(
                    sorted(holdings_with_value, key=lambda x: x["value"], reverse=True),
                    x="symbol",
                    y="value",
                    title="Position Values ($)",
                    color="symbol",
                    text_auto=".2s",
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)


def render_market_overview():
    """Render market overview page."""
    st.header("Market Overview")

    indices = st.session_state.market_data.get_market_indices() or {}

    def _fmt_index(name):
        d = indices.get(name, {})
        price = d.get("price", 0.0)
        chg_pct = d.get("change_percent", 0.0)
        value = f"{price:,.2f}" if price else "N/A"
        delta = f"{chg_pct:+.2f}%" if price else None
        return value, delta

    col1, col2, col3 = st.columns(3)

    with col1:
        v, d = _fmt_index("S&P 500")
        st.metric(label="S&P 500", value=v, delta=d)

    with col2:
        v, d = _fmt_index("Dow Jones")
        st.metric(label="Dow Jones", value=v, delta=d)

    with col3:
        v, d = _fmt_index("Nasdaq")
        st.metric(label="Nasdaq", value=v, delta=d)

    st.divider()

    st.subheader("Stock Quote Lookup")
    ticker = st.text_input("Enter ticker symbol", placeholder="AAPL")

    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            quote = st.session_state.market_data.get_stock_quote(ticker)
            if quote:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price", f"${quote['price']:.2f}")
                with col2:
                    st.metric("Change", f"${quote['change']:.2f}")
                with col3:
                    st.metric("% Change", f"{quote['change_percent']:.2f}%")

            try:
                hist = yf.Ticker(ticker).history(period="1mo")
                if not hist.empty:
                    hist = hist.reset_index()
                    fig_trend = px.line(
                        hist,
                        x="Date",
                        y="Close",
                        title=f"{ticker.upper()} — 30-Day Price History",
                        labels={"Close": "Price ($)", "Date": "Date"},
                    )
                    fig_trend.update_traces(line_color="#2196F3")
                    fig_trend.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_trend, use_container_width=True)
            except Exception:
                pass


def render_goal_planning():
    """Render goal planning page — routes through GoalPlanningAgent."""
    st.header("Financial Goal Planning")

    with st.form("goal_form"):
        goal_name = st.text_input("Goal Name", placeholder="e.g. Retire by 60, Buy a home")
        target_amount = st.number_input("Target Amount ($)", min_value=0.0, step=1000.0)
        timeline_years = st.number_input("Timeline (Years)", min_value=1, max_value=50, value=10)
        current_savings = st.number_input("Current Savings ($)", min_value=0.0, step=500.0)
        risk_tolerance = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"])

        submitted = st.form_submit_button("Create Goal Plan", type="primary")

    if submitted:
        errors = []
        if not goal_name.strip():
            errors.append("Goal Name is required.")
        if target_amount <= 0:
            errors.append("Target Amount must be greater than $0.")
        if errors:
            for msg in errors:
                st.error(msg)

    if submitted and goal_name.strip() and target_amount > 0:
        query = f"Help me plan for my goal: {goal_name}."
        with st.spinner("Building your personalized goal plan..."):
            output = st.session_state.workflow.process_query(
                query,
                context={
                    "goal_name": goal_name,
                    "target_amount": target_amount,
                    "timeline_years": timeline_years,
                    "current_savings": current_savings,
                    "risk_tolerance": risk_tolerance,
                    "investment_horizon": f"{timeline_years} years",
                },
            )
        st.markdown(output.get("final_response", "Could not generate a plan."))

        # Savings projection chart
        annual_return = {"conservative": 0.04, "moderate": 0.07, "aggressive": 0.10}[risk_tolerance]
        monthly_rate = annual_return / 12
        total_months = timeline_years * 12
        monthly_contrib = (
            (target_amount - current_savings * (1 + monthly_rate) ** total_months)
            / (((1 + monthly_rate) ** total_months - 1) / monthly_rate)
            if monthly_rate > 0 else
            (target_amount - current_savings) / total_months
        )
        monthly_contrib = max(monthly_contrib, 0)

        months = list(range(total_months + 1))
        balances = []
        balance = current_savings
        for m in months:
            balances.append(balance)
            balance = balance * (1 + monthly_rate) + monthly_contrib

        import pandas as pd
        proj_df = pd.DataFrame({
            "Month": months,
            "Balance ($)": balances,
        })
        proj_df["Year"] = proj_df["Month"] / 12

        st.divider()
        st.subheader("Savings Projection")
        fig_proj = px.line(
            proj_df,
            x="Year",
            y="Balance ($)",
            title=f"{goal_name} — Projected Savings Growth ({risk_tolerance.capitalize()} · {annual_return*100:.0f}% annual return)",
            labels={"Year": "Years from Now", "Balance ($)": "Projected Balance ($)"},
        )
        fig_proj.add_hline(
            y=target_amount,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Target ${target_amount:,.0f}",
            annotation_position="top left",
        )
        fig_proj.update_traces(line_color="#FF6B35")
        fig_proj.update_layout(hovermode="x unified")
        st.plotly_chart(fig_proj, use_container_width=True)
        st.caption(
            f"Assumes ${monthly_contrib:,.0f}/month contribution · "
            f"{annual_return*100:.0f}% annual return · starting balance ${current_savings:,.0f}"
        )





def main():
    """Main application entry point."""
    initialize_session_state()
    render_header()

    # Render sidebar settings
    render_sidebar_settings()

    # Create tab-based navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat",
        "📊 Portfolio Analysis",
        "📈 Market Overview",
        "🎯 Goal Planning",
    ])

    with tab1:
        render_chat_page()

    with tab2:
        render_portfolio_page()

    with tab3:
        render_market_overview()

    with tab4:
        render_goal_planning()


if __name__ == "__main__":
    main()
