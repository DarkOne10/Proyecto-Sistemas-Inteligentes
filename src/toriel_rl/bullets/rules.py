def damages(color: str, moving: bool, invulnerable: bool, focused: bool = False) -> bool:
    """Color rule: white always hurts, blue only while moving unfocused, orange only while still."""
    if invulnerable or color == "green":
        return False
    if color == "blue":
        return moving and not focused
    if color == "orange":
        return not moving
    return color == "white"
