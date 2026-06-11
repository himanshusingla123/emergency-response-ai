"""
observability/phoenix_setup.py
"""

import os
import atexit
import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from config import settings

logger = logging.getLogger(__name__)

# Enable OTEL debugging
logging.basicConfig(level=logging.INFO)

logging.getLogger(
    "opentelemetry.exporter.otlp.proto.http"
).setLevel(logging.DEBUG)

logging.getLogger(
    "opentelemetry.sdk.trace.export"
).setLevel(logging.DEBUG)

os.environ.setdefault("GOOGLE_ADK_DISABLE_TRACING", "true")

_tracer = None
_tracer_provider = None


def setup_phoenix():
    global _tracer, _tracer_provider

    if _tracer:
        return _tracer

    try:
        # Verify values loaded correctly
        logger.info(
            "PHOENIX_COLLECTOR_ENDPOINT=%s",
            settings.phoenix_collector_endpoint,
        )

        logger.info(
            "PHOENIX_API_KEY loaded=%s",
            bool(settings.phoenix_api_key),
        )

        # Official Phoenix setup
        from phoenix.otel import register

        _tracer_provider = register(
            project_name="google-rapid-hackathon",
        )

        # trace.set_tracer_provider(_tracer_provider)

        # Google GenAI auto instrumentation
        try:
            from openinference.instrumentation.google_genai import (
                GoogleGenAIInstrumentor,
            )

            GoogleGenAIInstrumentor().instrument(
                tracer_provider=_tracer_provider
            )

            logger.info(
                "GoogleGenAIInstrumentor enabled"
            )

        except Exception as e:
            logger.exception(
                "Failed to instrument Google GenAI: %s",
                e,
            )

        atexit.register(_force_flush)

        _tracer = trace.get_tracer(
            "emergency_system"
        )

        logger.info(
            "Phoenix tracing initialized successfully"
        )

        # Test span
        with _tracer.start_as_current_span(
            "phoenix_connection_test"
        ):
            logger.info(
                "Created Phoenix test span"
            )

        _force_flush()

        return _tracer

    except Exception as e:
        logger.exception(
            "Phoenix initialization failed: %s",
            e,
        )

        return trace.get_tracer(
            "emergency_system"
        )


def _force_flush():
    global _tracer_provider

    if _tracer_provider:
        logger.info(
            "Force flushing spans..."
        )

        try:
            _tracer_provider.force_flush(
                timeout_millis=10000
            )

            logger.info(
                "Flush complete"
            )

        except Exception:
            logger.exception(
                "Flush failed"
            )


tracer = setup_phoenix()