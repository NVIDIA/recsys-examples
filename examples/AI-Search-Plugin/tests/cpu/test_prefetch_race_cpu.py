"""Regression for the prefetch race seen on the effect line and the 32-GPU real model: the trainer pops a
prefetch entry while the background thread is still warming it. The waiter must always be released."""
import threading, time


def test_entry_object_released_even_if_popped():
    pref, lock = {}, threading.Lock()
    ent = {"event": threading.Event(), "held": None, "err": None, "rid": 1}
    with lock:
        pref["k"] = ent

    def bg():  # mimics ai_search_plugin.adapters.slime_prefetch _bg_loop: touches only its local entry object
        try:
            time.sleep(0.2)
            raise RuntimeError("fetch failed")
        except BaseException as e:
            ent["err"] = repr(e)
        finally:
            ent["event"].set()

    t = threading.Thread(target=bg); t.start()
    with lock:
        got = pref.pop("k", None)  # trainer takes the entry while bg is mid-fetch
    assert got is ent and got["event"].wait(timeout=5)
    t.join()
