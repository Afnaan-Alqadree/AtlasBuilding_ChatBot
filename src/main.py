from pathlib import Path
from .chatbot import Chatbot
from .data_profile import build_grounding_pack
build_grounding_pack(".", "atlas_grounding_pack.json")

BANNER = """
Atlas Chatbot 
  Basic:
    - list floors
    - find room 3.201
    - rooms on Floor 1
    - sensors of room 3.201
    - is room 3.201 occupied now
    - status now for Floor 1
    - free meeting rooms now on Floor 2
    - peak hours for Floor 2

  Advanced:
    - utilization floor 3
    - busiest rooms on floor 2
    - underused rooms on floor 4
    - compare floors 2 and 3

  Other:
    - db:check
    - help

Type 'exit' or 'quit' to leave.
"""

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def main():
    print(BANNER)
    bot = Chatbot(data_dir=project_root())

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        ans = bot.ask(q)
        print(ans.get("text", ""))

        df = ans.get("df")
        if df is not None:
            try:
                print(df.head(30).to_string(index=False))
            except Exception:
                print(df)

if __name__ == "__main__":
    main()
