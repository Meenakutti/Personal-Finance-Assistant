"""
Main entry point for the Personal Finance Assistant.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_streamlit_ui():
    """Run the Streamlit UI application."""
    logger.info("Starting Streamlit UI...")
    try:
        # Use subprocess to run streamlit as a module
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"], check=False)
    except Exception as e:
        logger.error(f"Error running Streamlit: {e}")
        print(f"Error: Could not start Streamlit. Make sure it's installed: pip install streamlit")


def run_cli():
    """Run in CLI mode for testing."""
    logger.info("Starting CLI mode...")
    from config.workflow import FinanceAssistantWorkflow

    workflow = FinanceAssistantWorkflow()

    print("\n" + "="*50)
    print("Personal Finance Assistant - CLI Mode")
    print("="*50)
    print("Type 'quit' to exit\n")

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            print("\nAssistant: Thinking...")
            output = workflow.process_query(query)
            print(f"Assistant: {output.get('final_response', 'No response')}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}\n")


def test_agents():
    """Test all agents."""
    logger.info("Testing agents...")
    from agents.registry import AgentRegistry

    registry = AgentRegistry()

    print("\n" + "="*50)
    print("Agent Testing")
    print("="*50)

    for agent_name in registry.list_agents():
        agent = registry.get_agent(agent_name)
        print(f"\n✓ {agent_name}: {agent.config.description}")

    # Test routing
    test_queries = [
        "What's the S&P 500 price?",
        "How should I diversify my portfolio?",
        "What are tax-advantaged retirement accounts?",
        "What's the best investment strategy?"
    ]

    print("\n" + "-"*50)
    print("Query Routing Tests")
    print("-"*50)

    for query in test_queries:
        routed = registry.route_query(query)
        print(f"\nQuery: {query}")
        print(f"Routed to: {', '.join(routed)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Personal Finance Assistant - AI-powered financial education"
    )

    parser.add_argument(
        "--mode",
        choices=["ui", "cli", "test"],
        default="ui",
        help="Run mode: ui (Streamlit), cli (Terminal), or test"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG"] = "true"
        logger.setLevel(logging.DEBUG)

    try:
        if args.mode == "ui":
            run_streamlit_ui()
        elif args.mode == "cli":
            run_cli()
        elif args.mode == "test":
            test_agents()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
