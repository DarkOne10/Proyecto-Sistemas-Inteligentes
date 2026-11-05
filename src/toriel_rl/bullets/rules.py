def damages(color: str, moving: bool, invulnerable: bool) -> bool:
    """Color rule: white always hurts, blue only while moving, orange only while still."""
    if invulnerable or color == "green":
        return False
    if color == "blue":
        return moving
    if color == "orange":
        return not moving
    return color == "white"
