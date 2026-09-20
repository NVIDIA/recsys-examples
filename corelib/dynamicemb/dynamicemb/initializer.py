import abc
from dataclasses import replace
from typing import Dict, List, Optional

from dynamicemb.dynamicemb_config import *
from dynamicemb.types import DEFAULT_UNIFORM_LOWER, DEFAULT_UNIFORM_UPPER, group_key_of
from dynamicemb_extensions import (
    CurandStateContext,
    const_init,
    const_init_table_params,
    debug_init,
    normal_init,
    normal_init_table_params,
    truncated_normal_init,
    truncated_normal_init_table_params,
    uniform_init,
    uniform_init_table_params,
)


def _with_default_bounds(
    args: DynamicEmbInitializerArgs,
) -> DynamicEmbInitializerArgs:
    """Bounds are the only parameters that may still be unset this late.

    The planner fills them from a table's num_embeddings, and nothing else has
    to. Returns a copy rather than filling them in place: the args belong to
    the caller's configuration, which stays as it was written.
    """
    if args.lower is not None and args.upper is not None:
        return args
    return replace(
        args,
        lower=DEFAULT_UNIFORM_LOWER if args.lower is None else args.lower,
        upper=DEFAULT_UNIFORM_UPPER if args.upper is None else args.upper,
    )


class MultiTableInitializer(abc.ABC):
    """The initializer of a fused module: one object covering all its tables.

    A fused module keeps several logical tables in one value buffer. They are
    grouped by initializer *mode* only -- see
    :meth:`~dynamicemb.dynamicemb_config.DynamicEmbTableOptions.get_grouped_key`
    -- so the mode is common but the parameters need not be, and routinely are
    not: an unbounded UNIFORM defaults to ``+/-sqrt(1 / num_embeddings)``, which
    is per table.

    When the tables do agree, or there is only one, each subclass calls its
    mode's plain kernel with the parameters as scalars. When they do not, it
    calls that mode's ``_table_params`` kernel with a
    ``[num_tables, num_params]`` tensor, and the kernel looks up the parameters
    of the table owning each row. Both kernels write the same multi-table
    buffer -- only where the parameters come from differs.
    """

    @staticmethod
    def create(
        args_list: List[DynamicEmbInitializerArgs],
        device: torch.device,
    ) -> "MultiTableInitializer":
        """The initializer for a module whose tables are configured like this."""
        if not args_list:
            raise ValueError("A module needs at least one table to initialize.")
        # Ask what the tables were grouped on rather than compare modes here,
        # so this cannot drift from what actually decided they may be fused.
        keys = {group_key_of(args) for args in args_list}
        if len(keys) != 1:
            raise ValueError(
                f"Tables of one module must agree on their initializer, got "
                f"{len(keys)} different ones: {keys}"
            )
        mode = args_list[0].mode
        if mode not in _INITIALIZERS:
            raise ValueError(f"Not supported initializer type: {mode}")

        args_list = [_with_default_bounds(args) for args in args_list]
        initializer_class = _INITIALIZERS[mode]
        rows = initializer_class.table_param_rows(args_list)
        table_params = None
        if any(row != rows[0] for row in rows):
            table_params = torch.tensor(rows, dtype=torch.float32, device=device)
        return initializer_class(args_list[0], table_params)

    def __init__(
        self,
        args: DynamicEmbInitializerArgs,
        table_params: Optional[torch.Tensor] = None,
    ):
        self._args = args
        self._table_params = table_params

    @staticmethod
    def table_param_rows(
        args_list: List[DynamicEmbInitializerArgs],
    ) -> List[List[float]]:
        """One row of parameters per table, in the order the kernel reads them."""
        return [[] for _ in args_list]

    def _table_ids_for_kernel(self, table_ids: Optional[torch.Tensor]):
        """The table each buffer row belongs to, which the lookup kernel needs."""
        if table_ids is None:
            raise ValueError(
                "This module's tables initialize differently, so the initializer "
                "needs the table id of every row it writes."
            )
        return table_ids

    @abc.abstractmethod
    def __call__(
        self,
        buffer: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
        table_ids: Optional[torch.Tensor],
        indices: torch.Tensor,
    ) -> None:
        """Initialize the rows of ``buffer`` that ``indices`` selects.

        Everything before ``indices`` runs alongside ``buffer``, one entry per
        row of it; ``indices`` comes last because it is the mask over those
        rows, not another thing to line up with them. The kernels check the
        lengths, so lining something up with ``indices`` instead is refused
        rather than read past the end of.
        """
        ...


class NormalInitializer(MultiTableInitializer):
    def __init__(self, args, table_params=None):
        super().__init__(args, table_params)
        self._curand_state = CurandStateContext()

    @staticmethod
    def table_param_rows(args_list):
        return [[args.mean, args.std_dev] for args in args_list]

    def __call__(self, buffer, keys, table_ids, indices) -> None:
        if self._table_params is None:
            normal_init(
                buffer,
                indices,
                self._curand_state,
                self._args.mean,
                self._args.std_dev,
            )
        else:
            normal_init_table_params(
                buffer,
                indices,
                self._curand_state,
                self._table_params,
                self._table_ids_for_kernel(table_ids),
            )


class TruncatedNormalInitializer(MultiTableInitializer):
    def __init__(self, args, table_params=None):
        super().__init__(args, table_params)
        self._curand_state = CurandStateContext()

    @staticmethod
    def table_param_rows(args_list):
        return [[args.mean, args.std_dev, args.lower, args.upper] for args in args_list]

    def __call__(self, buffer, keys, table_ids, indices) -> None:
        if self._table_params is None:
            truncated_normal_init(
                buffer,
                indices,
                self._curand_state,
                self._args.mean,
                self._args.std_dev,
                self._args.lower,
                self._args.upper,
            )
        else:
            truncated_normal_init_table_params(
                buffer,
                indices,
                self._curand_state,
                self._table_params,
                self._table_ids_for_kernel(table_ids),
            )


class UniformInitializer(MultiTableInitializer):
    def __init__(self, args, table_params=None):
        super().__init__(args, table_params)
        self._curand_state = CurandStateContext()

    @staticmethod
    def table_param_rows(args_list):
        return [[args.lower, args.upper] for args in args_list]

    def __call__(self, buffer, keys, table_ids, indices) -> None:
        if self._table_params is None:
            uniform_init(
                buffer,
                indices,
                self._curand_state,
                self._args.lower,
                self._args.upper,
            )
        else:
            uniform_init_table_params(
                buffer,
                indices,
                self._curand_state,
                self._table_params,
                self._table_ids_for_kernel(table_ids),
            )


class ConstantInitializer(MultiTableInitializer):
    @staticmethod
    def table_param_rows(args_list):
        return [[args.value] for args in args_list]

    def __call__(self, buffer, keys, table_ids, indices) -> None:
        if self._table_params is None:
            const_init(buffer, indices, self._args.value)
        else:
            const_init_table_params(
                buffer,
                indices,
                self._table_params,
                self._table_ids_for_kernel(table_ids),
            )


class DebugInitializer(MultiTableInitializer):
    # Fills a row from its key, so it takes no parameters at all and its tables
    # cannot disagree. Inherits table_param_rows, which then reports no
    # parameters for every table, so create never builds a tensor for it.
    def __call__(self, buffer, keys, table_ids, indices) -> None:
        debug_init(buffer, indices, keys)


_INITIALIZERS: Dict[DynamicEmbInitializerMode, type] = {
    DynamicEmbInitializerMode.NORMAL: NormalInitializer,
    DynamicEmbInitializerMode.TRUNCATED_NORMAL: TruncatedNormalInitializer,
    DynamicEmbInitializerMode.UNIFORM: UniformInitializer,
    DynamicEmbInitializerMode.CONSTANT: ConstantInitializer,
    DynamicEmbInitializerMode.DEBUG: DebugInitializer,
}
