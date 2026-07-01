from ktem.utils.guest_scope import default_file_selection, force_search_all_for_guest


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


def test_guest_submit_ignores_urls_and_web_search(monkeypatch):
    import pytest

    pytest.importorskip("gradio")

    import ktem.pages.chat.__init__ as chat_module
    from ktem.pages.chat import ChatPage

    page = object.__new__(ChatPage)

    def fail_index(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("guest URL input must not trigger indexing")

    page.first_indexing_url_fn = fail_index
    page.chat_control = object()
    monkeypatch.setattr(chat_module, "is_guest_user", lambda user_id: True)

    result = page.submit_msg(
        {"text": f"@{chat_module.WEB_SEARCH_COMMAND} https://example.edu"},
        [],
        "guest-id",
        {},
        "existing-conv",
        "Guest conversation",
        [],
        request=None,
    )

    assert result[-3] == "all"
    assert result[-2]["value"] == []
    assert result[-1] is None
    assert result[1][-1][0] == chat_module.DEFAULT_QUESTION
