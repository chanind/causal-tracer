import re


def guess_subject(prompt: str) -> str:
    match = re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)
    if not match:
        raise ValueError(f"Could not guess subject in prompt: {prompt}")
    return match[0].strip()
