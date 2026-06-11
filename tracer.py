"""
Quick Phoenix Cloud connectivity test.
Run: python test_phoenix.py
"""
import asyncio
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from config import settings


async def test_phoenix():
    base = settings.phoenix_collector_endpoint.rstrip("/")
    endpoint = base if base.endswith("/v1/traces") else f"{base}/v1/traces"

    print(f"Testing endpoint: {endpoint}")
    print(f"API key (first 8 chars): {settings.phoenix_api_key[:8]}...")

    # Setup
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                headers={"authorization": f"Bearer {settings.phoenix_api_key}"},
            )
        )
    )

    # Send a test span
    tracer = tracer_provider.get_tracer("phoenix-test")
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.message", "hello from test")
        span.set_attribute("test.status", "ok")
        print("Test span created ✓")

    # Force flush to send immediately
    tracer_provider.force_flush(timeout_millis=10_000)
    print("Spans flushed ✓")
    print("\nNow check Phoenix Cloud → your project 'google-rapid-hackathon'")
    print("If you see the span there, setup is correct!")


if __name__ == "__main__":
    asyncio.run(test_phoenix())