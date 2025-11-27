# rl_posttrain.py

PREFIX = "That is a great question. "
SUFFIX = " Let me know if you have any other questions."


def format_answer(text: str) -> str:
    """
    Wrap a raw answer with the required prefix/suffix format.
    """
    core = (text or "").strip()
    if not core:
        core = "I don't have an answer to that right now."

    if not core.endswith("."):
        core = core + "."

    return PREFIX + core + SUFFIX


if __name__ == "__main__":
    # Tiny manual test (does NOT run when FastAPI imports this file)
    raw = "this is a simple training text for this class assignment"
    print("RAW:", raw)
    print("FORMATTED:", format_answer(raw))
