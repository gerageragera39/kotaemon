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


def prepare_guest_chat_submission(
    chat_input_text: str,
    chat_history: list,
    default_question: str,
) -> tuple[str, list, None]:
    """Apply the guest-only submit policy without importing the Gradio UI."""
    if not chat_input_text and not chat_history:
        chat_input_text = default_question
    return chat_input_text, [], None
