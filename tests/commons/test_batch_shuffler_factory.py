"""
Unit tests for BatchShufflerFactory.
"""

import pytest
from commons.distributed.batch_shuffler import (
    BaseTaskBalancedBatchShuffler,
    IdentityBalancedBatchShuffler,
)
from commons.distributed.batch_shuffler_factory import (
    BatchShufflerFactory,
    register_batch_shuffler,
)


@pytest.fixture(autouse=True)
def restore_registry():
    """
    This function is a pytest fixture, used to restore the BatchShufflerFactory registry (_registry)
    before and after each test. Its main purpose is to prevent modifications to the registry in one
    test from affecting other tests, thus ensuring independence and isolation of test results.

    Usage scenario:
        When writing unit tests that modify global/class variables (such as registries or singletons),
        it's often necessary to save the original state before the test and restore it afterward,
        to avoid side effects across tests.

    Example:
        @pytest.fixture(autouse=True)
        def restore_registry():
            ...

        def test_register():
            BatchShufflerFactory.register("foo", FooShuffler)
            assert "foo" in BatchShufflerFactory._registry

        def test_unregister():
            BatchShufflerFactory.unregister("foo")
            assert "foo" not in BatchShufflerFactory._registry

        # Even if "foo" is registered in test_register, when test_unregister runs,
        # the state of "foo" in the registry is clean thanks to this fixture.
    """
    original_registry = BatchShufflerFactory._registry.copy()
    yield
    BatchShufflerFactory._registry = original_registry


def test_factory_cannot_be_instantiated():
    with pytest.raises(TypeError) as cm:
        BatchShufflerFactory()
    assert "should not be instantiated" in str(cm.value)
    assert "class methods" in str(cm.value).lower()


def test_register_valid_shuffler():
    class TestShuffler(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 0

    BatchShufflerFactory.register("test", TestShuffler)
    assert BatchShufflerFactory.is_registered("test")
    assert BatchShufflerFactory._registry["test"] is TestShuffler


def test_register_duplicate_name():
    class TestShuffler1(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 0

    class TestShuffler2(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 1

    BatchShufflerFactory.register("test", TestShuffler1)
    with pytest.raises(ValueError):
        BatchShufflerFactory.register("test", TestShuffler2)


def test_register_invalid_class():
    class NotAShuffler:
        pass

    with pytest.raises(TypeError):
        BatchShufflerFactory.register("invalid", NotAShuffler)


def test_unregister():
    class TestShuffler(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 0

    BatchShufflerFactory.register("test", TestShuffler)
    assert BatchShufflerFactory.is_registered("test")

    BatchShufflerFactory.unregister("test")
    assert not BatchShufflerFactory.is_registered("test")


def test_unregister_nonexistent():
    with pytest.raises(KeyError):
        BatchShufflerFactory.unregister("nonexistent")


def test_create_shuffler():
    class TestShuffler(BaseTaskBalancedBatchShuffler):
        def __init__(self, param1: int = 10, param2: str = "test"):
            self.param1 = param1
            self.param2 = param2

        def get_workloads(self, batch, *args, **kwargs):
            return self.param1

    BatchShufflerFactory.register("test", TestShuffler)

    shuffler1 = BatchShufflerFactory.create("test")
    assert isinstance(shuffler1, TestShuffler)
    assert shuffler1.param1 == 10
    assert shuffler1.param2 == "test"

    shuffler2 = BatchShufflerFactory.create("test", param1=20, param2="custom")
    assert shuffler2.param1 == 20
    assert shuffler2.param2 == "custom"


def test_create_nonexistent():
    with pytest.raises(KeyError) as cm:
        BatchShufflerFactory.create("nonexistent")
    assert "nonexistent" in str(cm.value)
    assert "Available shufflers" in str(cm.value)


def test_create_from_config():
    class TestShuffler(BaseTaskBalancedBatchShuffler):
        def __init__(self, param1: int, param2: str):
            self.param1 = param1
            self.param2 = param2

        def get_workloads(self, batch, *args, **kwargs):
            return self.param1

    BatchShufflerFactory.register("test", TestShuffler)

    config = {"type": "test", "param1": 42, "param2": "config_value"}
    shuffler = BatchShufflerFactory.create_from_config(config)

    assert isinstance(shuffler, TestShuffler)
    assert shuffler.param1 == 42
    assert shuffler.param2 == "config_value"


def test_create_from_config_missing_type():
    config = {"param1": 42}
    with pytest.raises(KeyError) as cm:
        BatchShufflerFactory.create_from_config(config)
    assert "type" in str(cm.value)


def test_list_available():
    class TestShuffler1(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 0

    class TestShuffler2(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 1

    BatchShufflerFactory.clear_registry()
    BatchShufflerFactory.register("test1", TestShuffler1)
    BatchShufflerFactory.register("test2", TestShuffler2)

    available = BatchShufflerFactory.list_available()
    assert len(available) == 2
    assert available["test1"] is TestShuffler1
    assert available["test2"] is TestShuffler2


def test_is_registered():
    class TestShuffler(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 0

    assert not BatchShufflerFactory.is_registered("test")
    BatchShufflerFactory.register("test", TestShuffler)
    assert BatchShufflerFactory.is_registered("test")


def test_clear_registry():
    class TestShuffler(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 0

    BatchShufflerFactory.register("test", TestShuffler)
    assert BatchShufflerFactory.is_registered("test")

    BatchShufflerFactory.clear_registry()
    assert not BatchShufflerFactory.is_registered("test")
    assert len(BatchShufflerFactory.list_available()) == 0


def test_decorator_registration():
    @register_batch_shuffler("decorated")
    class DecoratedShuffler(BaseTaskBalancedBatchShuffler):
        def get_workloads(self, batch, *args, **kwargs):
            return 0

    assert BatchShufflerFactory.is_registered("decorated")
    assert BatchShufflerFactory._registry["decorated"] is DecoratedShuffler
    assert DecoratedShuffler.__name__ == "DecoratedShuffler"


def test_identity_shuffler_registered():
    assert BatchShufflerFactory.is_registered("identity")
    shuffler = BatchShufflerFactory.create("identity")
    assert isinstance(shuffler, IdentityBalancedBatchShuffler)


@pytest.fixture(scope="module")
def import_actual_shufflers():
    hstu_available = False
    sid_gr_available = False

    try:
        import hstu.utils.hstu_batch_balancer  # noqa: F401

        hstu_available = True
    except ImportError:
        pass

    try:
        import sid_gr.utils.sid_batch_balancer  # noqa: F401

        sid_gr_available = True
    except ImportError:
        pass

    return hstu_available, sid_gr_available


def test_hstu_shuffler_registered(import_actual_shufflers):
    hstu_available, _ = import_actual_shufflers
    if not hstu_available:
        pytest.skip("HSTU module not available")
    assert BatchShufflerFactory.is_registered("hstu")
    shuffler = BatchShufflerFactory.create(
        "hstu", num_heads=16, head_dim=64, action_interleaved=True
    )
    assert shuffler.num_heads == 16
    assert shuffler.head_dim == 64
    assert shuffler.action_interleaved is True


def test_sid_gr_shuffler_registered(import_actual_shufflers):
    _, sid_gr_available = import_actual_shufflers
    if not sid_gr_available:
        pytest.skip("SID-GR module not available")
    assert BatchShufflerFactory.is_registered("sid_gr")
    shuffler = BatchShufflerFactory.create("sid_gr", num_heads=8, head_dim=128)
    assert shuffler.num_heads == 8
    assert shuffler.head_dim == 128
