"""
Goal Planning Agent: Assists with financial goal setting and planning.
"""

import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from agents.base_agent import BaseFinanceAgent, AgentConfig


class GoalPlanningAgent(BaseFinanceAgent):
    """
    Agent for financial goal planning and tracking.
    Helps users set, plan, and track financial goals.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="Goal Planning Agent",
                description="Assists with financial goal setting and planning",
                system_prompt="""You are an expert Goal Planning Agent helping users build actionable savings and investment plans.

ROLE:
1. GOAL CLARITY: Help users articulate SMART (Specific, Measurable, Achievable, Relevant, Time-bound) financial goals.
2. FEASIBILITY: Assess feasibility using the provided savings calculations (future-value formula, compound growth).
3. ROADMAP: Build concrete milestones based on calculated monthly savings requirements.
4. SCENARIOS: Present conservative (4% return), moderate (7%), and aggressive (10%) projections — return assumptions based on historical S&P 500 averages (source: Vanguard, "How to invest" guide; Ibbotson Associates long-run data).

ACCURACY GUIDELINES:
- Base all projections strictly on the calculated figures provided in the context — do not invent numbers.
- Cite your assumptions explicitly: e.g., "assuming a 7% annual return consistent with long-run diversified equity returns (Vanguard)".
- Acknowledge that actual returns vary and past performance does not guarantee future results (SEC Investor Bulletin).
- Do not recommend specific funds or securities — reference only asset classes (e.g., "low-cost broad-market index funds").
- Recommend professional advice (CFP) for large goals like retirement or home purchase.

DISCLAIMER: Goal projections are educational estimates only. Actual results will vary. Consult a Certified Financial Planner (CFP) before making significant financial commitments."""
            )
        super().__init__(config)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Help with financial goal planning.

        Args:
            query: Goal-related query
            context: User profile, financial situation, etc.

        Returns:
            Goal planning advice and actionable steps
        """
        self.add_to_history("user", query)

        goal_data = self._extract_goals_and_timeline(query, context)
        calculations = self._calculate_savings_requirements(goal_data)
        action_plan = self._create_action_plan(goal_data, calculations)
        tracking = self._generate_tracking_recommendations(goal_data, calculations)

        ctx = context or {}
        first_goal = goal_data["goals"][0] if goal_data.get("goals") else {}
        goals_analysis = calculations.get("goals_analysis", [{}])
        first_calc = goals_analysis[0] if goals_analysis else {}

        context_block = (
            "\n\n---\n"
            "USER-ENTERED VALUES (reference these exactly in your response):\n"
            f"- Goal: {ctx.get('goal_name', first_goal.get('type', 'N/A'))}\n"
            f"- Target Amount: ${first_goal.get('amount', 0):,.0f}\n"
            f"- Timeline: {first_goal.get('years', goal_data.get('timeline_years', 'N/A'))} years\n"
            f"- Current Savings: ${goal_data.get('current_savings', 0):,.0f}\n"
            f"- Risk Tolerance: {goal_data.get('risk_tolerance', 'moderate').capitalize()}\n"
            f"- Required Monthly Savings: ${first_calc.get('monthly_savings_needed', 0):,.2f}\n"
            f"- Expected Annual Return: {first_calc.get('expected_return', 0)}%\n"
            "\nCOMPUTED PROJECTIONS:\n"
            + json.dumps({
                "goal_data": goal_data,
                "calculations": calculations,
                "action_plan": action_plan,
                "tracking": tracking,
            }, default=str, indent=2)
        )
        response = self._call_llm(query, context_block)
        
        self.add_to_history("assistant", response)

        return response

    def _extract_goals_and_timeline(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract financial goals and timelines from structured context, falling back to query parsing."""
        ctx = context or {}
        goal_data = {
            "goals": [],
            "timeline_years": ctx.get("timeline_years", 5),
            "current_savings": ctx.get("current_savings", 0.0),
            "monthly_income": ctx.get("monthly_income", 0.0),
            "risk_tolerance": ctx.get("risk_tolerance", "moderate"),
        }

        # Prefer structured form data passed from the UI
        if ctx.get("target_amount") and ctx.get("timeline_years"):
            goal_data["goals"] = [{
                "amount": float(ctx["target_amount"]),
                "years": int(ctx["timeline_years"]),
                "type": ctx.get("goal_name", "general"),
            }]
            return goal_data

        # Fallback: keyword-match against the query text
        goal_keywords = {
            "retirement": {"amount": 1000000, "years": 30, "type": "retirement"},
            "home": {"amount": 500000, "years": 10, "type": "home"},
            "college": {"amount": 300000, "years": 18, "type": "education"},
            "emergency": {"amount": 15000, "years": 1, "type": "emergency"},
            "vacation": {"amount": 5000, "years": 1, "type": "vacation"},
        }
        for keyword, goal_info in goal_keywords.items():
            if keyword.lower() in query.lower():
                goal_data["goals"].append(goal_info)

        return goal_data

    def _calculate_savings_requirements(self, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate required savings and returns to achieve goals."""
        calculations = {
            "goals_analysis": [],
            "total_needed": 0.0,
            "monthly_savings_needed": 0.0,
            "annual_return_needed": 0.0
        }

        # Annual return rates by risk tolerance
        return_rates = {
            "conservative": 0.04,
            "moderate": 0.07,
            "aggressive": 0.10
        }
        
        annual_return = return_rates.get(goal_data["risk_tolerance"], 0.07)
        monthly_return = annual_return / 12

        for goal in goal_data["goals"]:
            years = goal.get("years", 5)
            target_amount = goal.get("amount", 50000)
            current_savings = goal_data["current_savings"]

            # Calculate future value needed
            months = years * 12
            
            # Calculate monthly savings needed using FV formula
            # FV = P(1 + r)^n + PMT * [((1 + r)^n - 1) / r]
            if monthly_return > 0:
                growth_factor = (1 + monthly_return) ** months
                fv_current = current_savings * growth_factor
                remaining_needed = max(0, target_amount - fv_current)
                
                if months > 0:
                    monthly_savings = remaining_needed / (((growth_factor - 1) / monthly_return)) if growth_factor > 1 else remaining_needed / months
                else:
                    monthly_savings = 0
            else:
                monthly_savings = target_amount / months if months > 0 else 0

            calculations["goals_analysis"].append({
                "goal_type": goal.get("type", "general"),
                "target_amount": target_amount,
                "years": years,
                "monthly_savings_needed": round(monthly_savings, 2),
                "expected_return": round(annual_return * 100, 1)
            })

            calculations["total_needed"] += target_amount
            calculations["monthly_savings_needed"] += monthly_savings

        calculations["monthly_savings_needed"] = round(calculations["monthly_savings_needed"], 2)
        calculations["annual_return_needed"] = round(annual_return * 100, 1)

        return calculations

    def _create_action_plan(self, goal_data: Dict[str, Any], calculations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create action plan with milestones."""
        action_plan = []
        
        today = datetime.now()
        
        for i, goal_analysis in enumerate(calculations["goals_analysis"]):
            years = goal_analysis["years"]
            target_amount = goal_analysis["target_amount"]
            monthly_savings = goal_analysis["monthly_savings_needed"]
            
            milestones = []
            
            # Create quarterly/yearly milestones
            for quarter in range(1, min(years * 4 + 1, 13)):  # Cap at 3 years
                milestone_date = today + timedelta(days=quarter * 90)
                progress_percent = (quarter / (years * 4)) * 100
                amount_accumulated = (monthly_savings * 3 * quarter)
                
                milestones.append({
                    "date": milestone_date.strftime("%Y-%m-%d"),
                    "quarter": quarter,
                    "progress_percent": min(100, progress_percent),
                    "target_amount": round(amount_accumulated, 2)
                })
            
            action_plan.append({
                "goal": goal_analysis["goal_type"],
                "target": f"${target_amount:,.0f}",
                "timeline": f"{years} years",
                "monthly_commitment": f"${monthly_savings:,.2f}",
                "milestones": milestones
            })
        
        return action_plan

    def _generate_tracking_recommendations(self, goal_data: Dict[str, Any], calculations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate progress tracking recommendations."""
        tracking = {
            "review_frequency": "Monthly",
            "key_metrics": [],
            "adjustment_triggers": []
        }

        # Key metrics based on risk tolerance
        if goal_data["risk_tolerance"] == "conservative":
            tracking["review_frequency"] = "Quarterly"
            tracking["key_metrics"] = [
                "Account balance vs. target",
                "Interest/dividend income",
                "Portfolio safety ratio"
            ]
        elif goal_data["risk_tolerance"] == "aggressive":
            tracking["review_frequency"] = "Monthly"
            tracking["key_metrics"] = [
                "Portfolio performance vs. benchmark",
                "Asset allocation drift",
                "Return rate achieved"
            ]
        else:
            tracking["key_metrics"] = [
                "Savings rate consistency",
                "Investment returns vs. target",
                "Account balance milestone achievement"
            ]

        # Add adjustment triggers
        tracking["adjustment_triggers"] = [
            "If actual savings fall 20% below monthly target",
            "If market performance drops 15% or more",
            "If personal income changes significantly",
            "Yearly review to rebalance portfolio"
        ]

        return tracking

    def _format_response(self, goal_data: Dict[str, Any], calculations: Dict[str, Any], 
                        action_plan: List[Dict[str, Any]], tracking: Dict[str, Any]) -> str:
        """Format comprehensive goal planning response."""
        response_parts = [
            "## Financial Goal Planning\n",
            f"**Risk Tolerance:** {goal_data['risk_tolerance'].capitalize()}\n",
            f"**Number of Goals:** {len(goal_data['goals'])}\n\n"
        ]

        # Goals Summary
        response_parts.append("### Goals Identified:\n")
        for analysis in calculations["goals_analysis"]:
            response_parts.append(
                f"- **{analysis['goal_type'].title()}:** ${analysis['target_amount']:,.0f} in {analysis['years']} years\n"
            )

        # Savings Requirements
        response_parts.append("\n### Savings Requirements:\n")
        response_parts.append(f"**Total Amount Needed:** ${calculations['total_needed']:,.2f}\n")
        response_parts.append(f"**Monthly Savings Required:** ${calculations['monthly_savings_needed']:,.2f}\n")
        response_parts.append(f"**Expected Annual Return:** {calculations['annual_return_needed']}%\n\n")

        # Action Plan
        response_parts.append("### Action Plan:\n")
        for plan in action_plan:
            response_parts.append(f"**{plan['goal'].title()}** ({plan['timeline']})\n")
            response_parts.append(f"- Target: {plan['target']}\n")
            response_parts.append(f"- Monthly Commitment: {plan['monthly_commitment']}\n")
            response_parts.append(f"- Key Milestones: {len(plan['milestones'])} checkpoints\n\n")

        # Progress Tracking
        response_parts.append("### Progress Tracking:\n")
        response_parts.append(f"**Review Frequency:** {tracking['review_frequency']}\n")
        response_parts.append("**Track These Metrics:**\n")
        for metric in tracking['key_metrics']:
            response_parts.append(f"- {metric}\n")
        response_parts.append("\n**Adjust Plan When:**\n")
        for trigger in tracking['adjustment_triggers']:
            response_parts.append(f"- {trigger}\n")

        response_parts.append("\n### Recommendations:\n")
        response_parts.append("1. Automate your monthly savings\n")
        response_parts.append("2. Review this plan quarterly\n")
        response_parts.append("3. Adjust contributions if income changes\n")
        response_parts.append("4. Consider tax-advantaged accounts\n")

        return "".join(response_parts)
