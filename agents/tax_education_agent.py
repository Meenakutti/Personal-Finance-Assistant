"""
Tax Education Agent: Explains tax concepts and account types.
"""

import json
from typing import Any, Dict, Optional, List
from agents.base_agent import BaseFinanceAgent, AgentConfig


class TaxEducationAgent(BaseFinanceAgent):
    """
    Agent for tax education and guidance.
    Explains tax concepts, account types, and strategies.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="Tax Education Agent",
                description="Explains tax concepts and account types",
                system_prompt="""You are an expert Tax Education Agent explaining U.S. tax concepts, retirement accounts, and tax-efficient strategies.

ROLE:
1. TAX CONCEPTS: Explain tax concepts with clear examples grounded in IRS publications.
2. ACCOUNT TYPES: Detail 401(k), IRA, Roth IRA, HSA, 529, and SEP-IRA using official IRS contribution limits and rules.
3. TAX STRATEGIES: Discuss general tax-efficiency strategies (tax-loss harvesting, asset location, Roth conversion ladders).
4. CAPITAL GAINS: Explain long-term vs. short-term treatment using current IRS rate schedules (IRS Topic No. 409).
5. DEDUCTIONS: Educate on standard deduction, itemized deductions, and common credits using IRS Publication 529 and related.

ACCURACY GUIDELINES:
- Cite the authoritative source for every rule or figure: e.g., "per IRS Publication 590-A (2024)", "per IRS Revenue Procedure 2023-34", "per IRS Topic No. 409".
- Use only the contribution limits and tax brackets provided in the knowledge base or widely published IRS data.
- State clearly when figures are for 2024 tax year and note they are adjusted annually.
- Never provide personalized tax advice, calculate a user's tax liability, or suggest specific tax strategies for their situation.
- Recommend consulting a CPA, enrolled agent, or tax attorney for personal situations.
- Acknowledge tax laws vary by state and that individual circumstances affect tax outcomes.

⚠️ DISCLAIMER: All information is educational and does not constitute tax advice. Tax laws are complex and situation-specific. Consult a qualified tax professional (CPA, enrolled agent, or tax attorney) before making tax-related decisions."""
            )
        super().__init__(config)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Provide tax education and guidance.

        Args:
            query: Tax-related question
            context: User profile, income level, etc.

        Returns:
            Tax education response with relevant information
        """
        self.add_to_history("user", query)

        # Primary: FAISS semantic search results injected by the LangGraph workflow
        rag_docs = (context or {}).get("documents", [])

        # Secondary: keyword-matched built-in tax knowledge (fills gaps when FAISS is thin)
        tax_info = self._retrieve_tax_information(query)
        account_info = self._discuss_account_types(query, context)
        concept_explanations = self._explain_tax_concepts(query, tax_info, context)

        # Only include account types that are relevant to the matched topics —
        # sending all 5 accounts for every question bloats the context unnecessarily.
        topic_to_accounts: Dict[str, List[str]] = {
            "401k": ["401k"],
            "ira":  ["traditional_ira", "roth_ira"],
            "hsa":  ["hsa"],
            "estimated taxes": ["seo_sep_ira"],
        }
        relevant_account_keys: set = set()
        for topic in tax_info.get("topics", []):
            relevant_account_keys.update(topic_to_accounts.get(topic, []))
        filtered_accounts = (
            {k: v for k, v in account_info.items() if k in relevant_account_keys}
            if relevant_account_keys else account_info
        )

        context_block = (
            "\n\nKnowledge base documents (FAISS semantic search results):\n"
            + json.dumps(
                [
                    {
                        "title": d.get("title", ""),
                        "category": d.get("category", ""),
                        "content": d.get("content", ""),
                        "source": d.get("metadata", {}).get("source", d.get("source", "")),
                    }
                    for d in rag_docs
                ],
                default=str,
                indent=2,
            )
            + "\n\nBuilt-in tax knowledge (keyword-matched):\n"
            + json.dumps({"tax_topics": tax_info, "account_types": filtered_accounts}, default=str, indent=2)
            + "\n\nPre-computed concept explanations (exact figures for 2024):\n"
            + json.dumps(concept_explanations, default=str, indent=2)
        )
        response = self._call_llm(query, context_block)

        self.add_to_history("assistant", response)

        return response

    def _retrieve_tax_information(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant tax information from knowledge base."""
        tax_data = {
            "topics": [],
            "relevant_concepts": [],
            "sources": []
        }

        # Tax-related knowledge base
        tax_knowledge = {
            "income tax": {
                "concepts": ["Progressive tax system", "Tax brackets", "Marginal vs effective tax rate", "Standard deduction"],
                "description": "Income tax is a direct tax on personal and corporate income. The U.S. uses a progressive tax system with tax brackets that determine how much tax you owe based on income level."
            },
            "capital gains": {
                "concepts": ["Long-term gains", "Short-term gains", "Tax rates", "Loss harvesting"],
                "description": "Capital gains are profits from selling investments. Long-term gains (held >1 year) are taxed at 0%, 15%, or 20%. Short-term gains are taxed as ordinary income."
            },
            "deduction": {
                "concepts": ["Standard deduction", "Itemized deductions", "Deduction limits", "AGI"],
                "description": "Deductions reduce your taxable income. You can either take the standard deduction or itemize deductions. The standard deduction for 2024 is $13,850 (single) or $27,700 (married filing jointly)."
            },
            "401k": {
                "concepts": ["Employer match", "Contribution limits", "Early withdrawal penalty", "Roth vs Traditional"],
                "description": "A 401(k) is an employer-sponsored retirement plan. Contributions are often tax-deductible, grow tax-deferred, and many employers offer matching contributions."
            },
            "ira": {
                "concepts": ["Traditional IRA", "Roth IRA", "Contribution limits", "Income limits"],
                "description": "Individual Retirement Accounts (IRAs) are personal retirement savings accounts. Traditional IRAs offer tax deductions, while Roth IRAs offer tax-free growth."
            },
            "hsa": {
                "concepts": ["Health Savings Account", "Triple tax advantage", "Contribution limits", "Medical expenses"],
                "description": "Health Savings Accounts (HSAs) offer triple tax advantages: tax-deductible contributions, tax-free growth, and tax-free withdrawals for medical expenses."
            },
            "tax credits": {
                "concepts": ["Earned Income Tax Credit", "Child Tax Credit", "Refundable credits", "Non-refundable credits"],
                "description": "Tax credits directly reduce taxes owed. Some credits are refundable (you get money back) while others only reduce tax liability."
            },
            "estimated taxes": {
                "concepts": ["Quarterly payments", "Self-employment tax", "Penalty avoidance", "Payment schedule"],
                "description": "Estimated taxes are quarterly tax payments for self-employed individuals or those with investment income. Missing payments can result in penalties."
            }
        }

        # Match query to tax topics
        query_lower = query.lower()
        for topic, data in tax_knowledge.items():
            if topic in query_lower or any(concept.lower() in query_lower for concept in data["concepts"]):
                tax_data["topics"].append(topic)
                tax_data["relevant_concepts"].extend(data["concepts"])
                tax_data["sources"].append({
                    "topic": topic,
                    "description": data["description"],
                    "concepts": data["concepts"]
                })

        # If no specific match, provide general tax information
        if not tax_data["sources"]:
            tax_data["sources"] = [tax_knowledge["income tax"]]
            tax_data["topics"] = ["income tax"]

        return tax_data

    def _explain_tax_concepts(self, query: str, tax_info: Dict[str, Any],
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Return only the concept explanations relevant to the matched tax topics."""
        all_explanations: Dict[str, str] = {}

        # Tax bracket explanation
        all_explanations["tax_brackets"] = """
**Tax Brackets:**
The U.S. uses a progressive tax system with tax brackets. This means different portions of your income are taxed at different rates, not your entire income at the top rate.

Example (2024 Single Filer):
- 10% on income up to $11,600
- 12% on income from $11,600 to $47,150
- 22% on income from $47,150 to $100,525
- 24% on income above $100,525

If you earn $60,000:
- First $11,600 × 10% = $1,160
- Next $35,550 × 12% = $4,266
- Next $12,850 × 22% = $2,827
- Total tax: $8,253 (effective rate: 13.8%)
"""

        all_explanations["capital_gains"] = """
**Capital Gains Tax Implications:**
How your investments are taxed depends on how long you held them:

Long-term Capital Gains (held > 1 year):
- 0% rate: $0-$46,025 (single)
- 15% rate: $46,025-$518,900 (single)
- 20% rate: $518,900+ (single)

Short-term Capital Gains (held ≤ 1 year):
- Taxed as ordinary income (highest rate 37%)

Strategy: Hold investments longer than 1 year to benefit from lower long-term capital gains rates.
"""

        all_explanations["deductions"] = """
**Standard vs. Itemized Deductions:**

Standard Deduction (2024):
- Single: $13,850
- Married Filing Jointly: $27,700
- Head of Household: $20,800

Itemized Deductions can include:
- Mortgage interest
- State and local taxes (up to $10,000)
- Charitable contributions
- Medical expenses (exceeding 7.5% of AGI)

Strategy: If itemized deductions exceed the standard deduction, itemize. Otherwise, take the standard deduction.
"""

        all_explanations["tax_efficiency"] = """
**Tax Efficiency Strategies:**

1. **Tax-Advantaged Accounts:**
   - 401(k): Up to $23,500/year (2024)
   - Traditional IRA: Up to $7,000/year
   - Roth IRA: Up to $7,000/year (income limits apply)
   - HSA: Up to $4,150/year (self-only coverage)

2. **Tax-Loss Harvesting:**
   - Sell losing positions to offset gains
   - Can deduct up to $3,000 of net losses against income
   - Carry forward unlimited losses

3. **Asset Location:**
   - Hold tax-inefficient assets (bonds, REITs) in retirement accounts
   - Hold tax-efficient assets (growth stocks, index funds) in taxable accounts
"""

        all_explanations["self_employment"] = """
**Self-Employment Tax:**
Self-employed individuals must pay both income tax and self-employment tax (Social Security + Medicare).

Self-Employment Tax Rate: 15.3%
- 12.4% for Social Security (on net earnings up to $168,600)
- 2.9% for Medicare (on all net earnings)

You can deduct half of your self-employment tax from gross income, reducing your tax burden.

Common Deductions for Self-Employed:
- Home office expenses
- Equipment and supplies
- Marketing and advertising
- Professional development
- Vehicle expenses (actual or standard mileage rate)
"""

        # Map each matched topic to the concept key(s) that explain it
        topic_to_keys: Dict[str, List[str]] = {
            "income tax":      ["tax_brackets"],
            "capital gains":   ["capital_gains"],
            "deduction":       ["deductions"],
            "401k":            ["tax_efficiency"],
            "ira":             ["tax_efficiency"],
            "hsa":             ["tax_efficiency"],
            "tax credits":     ["tax_brackets"],
            "estimated taxes": ["self_employment"],
        }
        relevant_keys: set = set()
        for topic in tax_info.get("topics", []):
            relevant_keys.update(topic_to_keys.get(topic, []))

        # Fall back to the most general concept if nothing matched
        if not relevant_keys:
            relevant_keys = {"tax_brackets"}

        return {k: v for k, v in all_explanations.items() if k in relevant_keys}

    def _discuss_account_types(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Discuss various retirement and savings account types."""
        account_types = {
            "401k": {
                "name": "401(k) Plan",
                "type": "Employer-Sponsored Retirement Plan",
                "contribution_limit_2024": "$23,500 (+ $7,500 catch-up if 50+)",
                "tax_treatment": "Traditional: Tax-deductible contributions, tax-deferred growth, taxed on withdrawal | Roth: Tax-free contributions, tax-free growth",
                "early_withdrawal": "59½ without penalty (some exceptions)",
                "employer_match": "Often includes employer matching contributions",
                "advantages": [
                    "High contribution limits",
                    "Potential employer match (free money)",
                    "Tax-deferred growth",
                    "Loan options available"
                ],
                "disadvantages": [
                    "Employer-dependent",
                    "Limited investment choices",
                    "Early withdrawal penalties"
                ]
            },
            "traditional_ira": {
                "name": "Traditional IRA",
                "type": "Individual Retirement Account",
                "contribution_limit_2024": "$7,000 (+ $1,000 catch-up if 50+)",
                "tax_treatment": "Tax-deductible contributions (subject to income limits), tax-deferred growth, taxed on withdrawal",
                "early_withdrawal": "59½ without penalty",
                "advantages": [
                    "Tax deduction for contributions",
                    "Tax-deferred growth",
                    "Can contribute even without earned income",
                    "Flexible investment options"
                ],
                "disadvantages": [
                    "Income limits for tax deduction",
                    "Required Minimum Distributions (RMDs) at 73",
                    "Early withdrawal penalties",
                    "Taxes on all withdrawals"
                ]
            },
            "roth_ira": {
                "name": "Roth IRA",
                "type": "Individual Retirement Account",
                "contribution_limit_2024": "$7,000 (+ $1,000 catch-up if 50+)",
                "tax_treatment": "After-tax contributions, tax-free growth, tax-free withdrawals in retirement",
                "early_withdrawal": "Contributions anytime tax/penalty-free; earnings at 59½",
                "advantages": [
                    "Tax-free growth and withdrawals",
                    "No Required Minimum Distributions",
                    "Can withdraw contributions anytime",
                    "Excellent for long-term growth",
                    "No income limits for conversions"
                ],
                "disadvantages": [
                    "No upfront tax deduction",
                    "Income limits for direct contributions",
                    "Contribution limits"
                ]
            },
            "hsa": {
                "name": "Health Savings Account",
                "type": "Medical Savings Account",
                "contribution_limit_2024": "$4,150 (self-only) or $8,300 (family coverage)",
                "tax_treatment": "Tax-deductible contributions, tax-free growth, tax-free withdrawals for qualified medical expenses",
                "early_withdrawal": "Medical expenses anytime; other uses at 65 (taxed like IRA)",
                "advantages": [
                    "Triple tax advantage (deductible, grows tax-free, tax-free withdrawals)",
                    "Can invest contributions",
                    "Unused funds roll over",
                    "Portable (not employer-dependent)"
                ],
                "disadvantages": [
                    "Must have high-deductible health plan (HDHP)",
                    "Limited to medical expenses",
                    "Requires tracking of expenses"
                ]
            },
            "seo_sep_ira": {
                "name": "SEP-IRA or Solo 401(k)",
                "type": "Self-Employed Retirement Plans",
                "contribution_limit_2024": "SEP: 25% of net self-employment income (max $69,000) | Solo 401(k): Up to $69,000",
                "tax_treatment": "Tax-deductible contributions, tax-deferred growth, taxed on withdrawal",
                "early_withdrawal": "59½ without penalty",
                "advantages": [
                    "High contribution limits",
                    "Simple to set up (SEP)",
                    "Solo 401(k) allows loans",
                    "Tax-deferred growth"
                ],
                "disadvantages": [
                    "Self-employed only",
                    "RMDs required at 73",
                    "Withdrawal penalties before 59½"
                ]
            }
        }

        return account_types

