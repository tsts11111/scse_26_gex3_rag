"""Non-interactive verification of the completed RAG agent."""

import re

from policy_loader import load_policy_documents
from agent import build_agent

documents = load_policy_documents()
agent = build_agent(documents)

QUESTIONS = [
    "What exact phone number should I call for the University IT Service Desk?",
    "My account has been locked for 30 minutes after failed logins. What now?",
    "Can I share my Campus-Secure password with a visiting friend?",
]

PHONE_SHAPE = re.compile(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\d{5,}")

for q in QUESTIONS:
    a = agent.invoke(q)
    print("Q:", q)
    print("A:", a)
    print("---")

print("=== Target-question anti-hallucination check ===")
target_answer = agent.invoke(QUESTIONS[0])
has_number = bool(PHONE_SHAPE.search(target_answer))
says_unknown = any(
    word in target_answer.lower()
    for word in [
        "not provide",
        "not available",
        "does not",
        "do not have",
        "not found",
        "no phone",
        "not listed",
        "not include",
        "unavailable",
        "cannot provide",
        "can't provide",
    ]
)
print("Contains an invented phone number:", has_number)
print("States the number is not available/provided:", says_unknown)
print("PASS" if (not has_number and says_unknown) else "NEEDS WORK")
