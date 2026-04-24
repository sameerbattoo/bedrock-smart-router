"""OpenTelemetry Integration — distributed tracing and metrics.

The router emits OTEL spans and metrics for every routing decision
when enabled.  Works with any OTEL-compatible backend: AWS X-Ray
(via ADOT), Jaeger, Datadog, Grafana Tempo, Honeycomb, etc.

Install:
    pip install bedrock-smart-router[otel]

Demonstrates:
  - Enabling OTEL via config
  - Viewing traces with console exporter (for demo)
  - Metrics emitted per request
  - Combining OTEL with CloudWatch and callbacks
  - Using with AWS Distro for OpenTelemetry (ADOT)
"""

# ═══════════════════════════════════════════════════════════════════
# Example 1: Enable OTEL via config — simplest setup
# ═══════════════════════════════════════════════════════════════════
# Just set otel_enabled: true.  The router creates a tracer and meter
# using the opentelemetry API.  Your OTEL collector picks them up.

from bedrock_smart_router import BedrockRouter

router = BedrockRouter.create({
    "strategy": "balanced",
    "observability": {
        "log_decisions": True,
        "otel_enabled": True,
        "otel_service_name": "my-app",
    },
})

# Every converse() call now emits:
#   - OTEL metrics: bedrock_router.requests, bedrock_router.latency,
#     bedrock_router.cost, bedrock_router.ttft, bedrock_router.cache_hits,
#     bedrock_router.fallbacks, bedrock_router.errors
#   - Labels: model, strategy, complexity

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "What is S3?"}]}],
)
d = response["routing_decision"]
print(f"Model: {d.selected_model}, OTEL enabled: {router.otel.enabled}")


# ═══════════════════════════════════════════════════════════════════
# Example 2: Console exporter — see traces in your terminal
# ═══════════════════════════════════════════════════════════════════
# Useful for development and debugging.  In production, replace with
# OTLP exporter pointing to your collector.

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

# Set up trace exporter (prints spans to console)
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(trace_provider)

# Set up metrics exporter (prints metrics to console every 5s)
from opentelemetry import metrics
metric_reader = PeriodicExportingMetricReader(
    ConsoleMetricExporter(),
    export_interval_millis=5000,
)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# Now create the router — it picks up the global OTEL providers
router = BedrockRouter.create({
    "observability": {
        "otel_enabled": True,
        "otel_service_name": "demo-app",
    },
})

response = router.converse(
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
print(f"\nCheck console output above for OTEL span and metrics ^")


# ═══════════════════════════════════════════════════════════════════
# Example 3: OTEL metrics emitted per request
# ═══════════════════════════════════════════════════════════════════
# The router emits these metrics with {model, strategy, complexity} labels:
#
# | Metric                        | Type      | Unit |
# |-------------------------------|-----------|------|
# | bedrock_router.requests       | Counter   | 1    |
# | bedrock_router.latency        | Histogram | ms   |
# | bedrock_router.ttft           | Histogram | ms   |
# | bedrock_router.cost           | Counter   | USD  |
# | bedrock_router.cache_hits     | Counter   | 1    |
# | bedrock_router.fallbacks      | Counter   | 1    |
# | bedrock_router.errors         | Counter   | 1    |
#
# These are standard OTEL metrics — your backend (Prometheus, CloudWatch,
# Datadog) can aggregate, alert, and dashboard on them.


# ═══════════════════════════════════════════════════════════════════
# Example 4: Combine OTEL + CloudWatch + custom callbacks
# ═══════════════════════════════════════════════════════════════════
# All three observability channels work simultaneously.

events = []

router = BedrockRouter.create(
    {
        "observability": {
            "log_decisions": True,           # Python logging
            "cloudwatch_enabled": True,      # CloudWatch PutMetricData
            "cloudwatch_namespace": "MyApp",
            "otel_enabled": True,            # OpenTelemetry
            "otel_service_name": "my-app",
        },
    },
    callbacks=[lambda e: events.append(e.decision.selected_model)],
)

# One converse() call → all four channels fire:
# 1. Python log: "routing_decision model=... strategy=... cost=..."
# 2. CloudWatch: 7 custom metrics with dimensions
# 3. OTEL: spans + 7 metrics with labels
# 4. Callback: events.append(model_name)


# ═══════════════════════════════════════════════════════════════════
# Example 5: AWS Distro for OpenTelemetry (ADOT) setup
# ═══════════════════════════════════════════════════════════════════
# In production on AWS, use ADOT to send traces to X-Ray and metrics
# to CloudWatch (or Prometheus via AMP).
#
# 1. Deploy the ADOT collector as a sidecar or daemon:
#    https://aws-otel.github.io/docs/getting-started/collector
#
# 2. Set the OTLP endpoint:
#    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
#
# 3. Install the OTLP exporter:
#    pip install opentelemetry-exporter-otlp
#
# 4. Configure in your app:
#
#    from opentelemetry import trace
#    from opentelemetry.sdk.trace import TracerProvider
#    from opentelemetry.sdk.trace.export import BatchSpanProcessor
#    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
#
#    trace_provider = TracerProvider()
#    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
#    trace.set_tracer_provider(trace_provider)
#
# 5. Create the router with OTEL enabled:
#    router = BedrockRouter.create({"observability": {"otel_enabled": True}})
#
# Traces will appear in AWS X-Ray, metrics in CloudWatch or AMP.


# ═══════════════════════════════════════════════════════════════════
# Example 6: Streaming with OTEL — TTFT metric
# ═══════════════════════════════════════════════════════════════════
# converse_stream() emits the same OTEL metrics as converse(), plus
# the bedrock_router.ttft histogram for time-to-first-token.

from bedrock_smart_router import RoutingConfig

router = BedrockRouter.create({
    "observability": {"otel_enabled": True, "otel_service_name": "stream-demo"},
})

for event in router.converse_stream(
    messages=[{"role": "user", "content": [{"text": "Write a haiku."}]}],
    routing=RoutingConfig(preset="speed"),
):
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"].get("text", ""), end="", flush=True)
    elif "routing_decision" in event:
        d = event["routing_decision"]
        print(f"\n[TTFT: {d.ttft_ms}ms → emitted as bedrock_router.ttft metric]")
