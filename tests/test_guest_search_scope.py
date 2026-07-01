from ktem.utils.commands import WEB_SEARCH_COMMAND
from ktem.utils.conversation import get_file_names_regex, get_urls
from ktem.utils.guest_scope import (
    default_file_selection,
    force_search_all_for_guest,
    prepare_guest_chat_submission,
)


def test_guest_selector_is_forced_to_search_all():
    resolved = force_search_all_for_guest(
        ["disabled", ["guest-owned-file"], -1],
        "guest-id",
        guest_user=True,
    )

    assert resolved == ["all", [], "guest-id"]


def test_file_management_default_remains_disabled():
    assert default_file_selection(user_management=True) == ("disabled", [], -1)


def test_normal_user_selector_mode_is_preserved():
    resolved = force_search_all_for_guest(
        ["select", ["user-file"], "stale-user"],
        "normal-user",
        guest_user=False,
    )

    assert resolved == ["select", ["user-file"], "normal-user"]


def test_retriever_rebinds_hidden_selector_to_callback_user():
    from types import SimpleNamespace

    from ktem.index.file.index import FileIndex

    class Selector:
        def resolve_selection_for_user(self, selected, user_id):
            assert selected == ["disabled", [], -1]
            assert user_id == "guest-id"
            return ["all", [], user_id]

        def get_selected_ids(self, selected):
            assert selected == ["all", [], "guest-id"]
            return ["admin-document"]

    class Retriever:
        @classmethod
        def get_pipeline(cls, settings, config, selected_ids):
            assert selected_ids == ["admin-document"]
            return SimpleNamespace()

    index = object.__new__(FileIndex)
    index.id = 1
    index.config = {}
    index._selector_ui = Selector()
    index._retriever_pipeline_cls = [Retriever]
    index._resources = {"Source": object(), "Index": object()}
    index._vs = object()
    index._docstore = object()
    index._fs_path = object()

    pipelines = index.get_retriever_pipelines({}, "guest-id", ["disabled", [], -1])

    assert len(pipelines) == 1
    assert pipelines[0].user_id == "guest-id"


def test_guest_submit_ignores_urls_and_web_search():
    text = f'@"{WEB_SEARCH_COMMAND}" https://example.edu'
    file_names, text = get_file_names_regex(text)
    urls, text = get_urls(text)

    resolved_text, file_ids, used_command = prepare_guest_chat_submission(
        text, [], "Default question"
    )

    assert file_names == [WEB_SEARCH_COMMAND]
    assert urls == ["https://example.edu"]
    assert resolved_text == "Default question"
    assert file_ids == []
    assert used_command is None
