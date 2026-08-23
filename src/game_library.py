# Should be in same directory as `my_app.py`

# Note: type annotations are *not* required by hydra-zen

__all__ = ["inventory", "Character"]


class Character:
    def __init__(self, name: str, level: int = 1, inventory=None):
        self.name = name
        self.level = level
        self.inventory = inventory

    def __repr__(self):
        out = ""
        out += f"{self.name}, "
        out += f"lvl: {self.level}, "
        out += f"has: {self.inventory}"
        return out


def inventory(gold: int, weapon: str, costume: str):
    return {"gold": gold, "weapon": weapon, "costume": costume}