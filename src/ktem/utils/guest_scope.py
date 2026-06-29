def default_file_selection(user_management: bool):
    """Keep file search disabled until an authenticated user chooses a scope."""
    return "disabled", [], -1 if user_management else 1


def force_search_all_for_guest(components, user_id, guest_user: bool):
    """Bind selector state to its callback user and force guests to search all."""
    resolved = list(components or [])
    resolved.extend([None] * (3 - len(resolved)))
    if guest_user:
        return ["all", [], user_id]

    resolved[2] = user_id
    return resolved
