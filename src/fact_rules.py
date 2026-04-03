KNOWN_FACTS = {
    "prime minister of india": "narendra modi",
    "president of india": "droupadi murmu",
    "capital of india": "new delhi",
    "currency of india": "rupee",
    "chief minister of tamil nadu": "m.k. stalin"
}

def verify_known_facts(text):
    """
    Checks if a known fact block is explicitly questioned in the text.
    If the correct answer exists, it flags it as real. If not, it flags it as fake.
    """
    text_lower = text.lower()
    for topic, answer in KNOWN_FACTS.items():
        if topic in text_lower:
            if answer in text_lower:
                return "REAL", f"The {topic} is {answer.title()}."
            else:
                return "FAKE", f"The actual {topic} is {answer.title()}."
    return None, None
