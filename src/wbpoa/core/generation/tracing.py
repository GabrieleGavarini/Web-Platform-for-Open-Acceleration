from __future__ import annotations

from typing import TYPE_CHECKING

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from structlog import get_logger

from wbpoa.core.settings import LangfuseSettings

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

log = get_logger()


class LangfuseTracer:
    """
    A Langfuse client and LangChain callback handler built from LangfuseSettings (Remark 17).

    A no-op when `LangfuseSettings.public_key`/`secret_key` are unset, matching that class's
    "optional: leave unset to run without tracing" contract -- callers check `enabled` (or just
    use `config`, which is None either way) rather than branching on which fields are set.

    Attributes:
        client: The underlying Langfuse client, or None when tracing is disabled.

    """

    def __init__(self, langfuse_settings: LangfuseSettings) -> None:
        """
        Args:
            langfuse_settings: public_key/secret_key/host, as already defined on CoreSettings.langfuse.

        """
        self.client: Langfuse | None = None
        self._config: RunnableConfig | None = None
        if langfuse_settings.public_key is None or langfuse_settings.secret_key is None:
            return

        try:
            public_key = langfuse_settings.public_key.get_secret_value()
            client = Langfuse(
                public_key=public_key,
                secret_key=langfuse_settings.secret_key.get_secret_value(),
                host=langfuse_settings.host,
            )
            config: RunnableConfig = {"callbacks": [CallbackHandler(public_key=public_key)]}
        except Exception:
            log.warning("langfuse_tracer_init_failed", exc_info=True)
            return

        self.client = client
        self._config = config

    @property
    def enabled(self) -> bool:
        """Whether tracing is active, i.e. both Langfuse keys are configured."""
        return self.client is not None

    @property
    def config(self) -> RunnableConfig | None:
        """A LangChain run config attaching the Langfuse callback handler, or None when disabled."""
        return self._config
