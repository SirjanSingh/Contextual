import threading

from repo_aware_ai.conversation import ConversationHistory


def test_history_buffers_to_max_turns():
    h = ConversationHistory(max_turns=3)
    for i in range(5):
        h.add_turn(f"q{i}", f"a{i}", [f"src{i}.py:0-10"])
    assert len(h) == 3
    # Most recent should win.
    assert h.get_last_question() == "q4"


def test_history_get_context_is_empty_initially():
    h = ConversationHistory()
    assert h.is_empty
    assert h.get_context() == ""


def test_history_get_context_renders_recent_turns():
    h = ConversationHistory(max_turns=2)
    h.add_turn("first?", "yes", [])
    h.add_turn("second?", "no", [])
    ctx = h.get_context()
    assert "first?" in ctx
    assert "second?" in ctx
    assert "yes" in ctx
    assert "no" in ctx


def test_history_clear():
    h = ConversationHistory()
    h.add_turn("q", "a", [])
    h.clear()
    assert h.is_empty


def test_history_thread_safe_under_contention():
    h = ConversationHistory(max_turns=100)

    def writer(start: int):
        for i in range(start, start + 100):
            h.add_turn(f"q{i}", f"a{i}", [])

    threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The buffer is bounded; no exceptions and length is exactly max_turns.
    assert len(h) == 100
