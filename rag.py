"""
Main entry point for the RAG exercise.

Run:
    python rag.py
Then type a question at the prompt (or press Enter on an empty line to quit).

The required behaviour:
    > What exact phone number should I call for the University IT Service Desk?
    The model must answer that the available university IT information does not
    provide the number. It must NOT invent one.
"""

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from policy_loader import load_policy_documents

documents = load_policy_documents()

from agent import build_agent

agent = build_agent(documents)

# Questions that must behave correctly. The first is the graded target;
# the others show the same agent still answers facts that ARE documented.
DEMO_QUESTIONS = (
    "What exact phone number should I call for the University IT Service Desk?",
    "My account has been locked for 30 minutes after failed logins. What now?",
    "Can I share my Campus-Secure password with a visiting friend?",
)


def main():
    print(f"RAG agent ready ({len(documents)} policy documents indexed).\n")

    print("=== Graded target question ===")
    target = DEMO_QUESTIONS[0]
    print("Q:", target)
    print("A:", agent.invoke(target), "\n")

    print("=== Sanity checks (facts that ARE in the policies) ===")
    for question in DEMO_QUESTIONS[1:]:
        print("Q:", question)
        print("A:", agent.invoke(question), "\n")

    print("=== Interactive mode (empty line to quit) ===")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print("A:", agent.invoke(question), "\n")


if __name__ == "__main__":
    main()
