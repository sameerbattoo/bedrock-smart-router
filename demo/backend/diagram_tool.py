# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Strands tool for generating AWS architecture diagrams.

Replaces the deprecated awslabs.aws-diagram-mcp-server with a direct
integration using the `diagrams` Python package (mingrammer/diagrams).

The tool exposes three functions to the agent:
- generate_diagram: Execute Python diagram code and return the PNG path
- get_diagram_examples: Show example code for various diagram types
- list_diagram_icons: List available icon providers/services for discovery
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)

# Output directory for generated diagrams
DIAGRAM_DIR = Path("/tmp/generated-diagrams")
DIAGRAM_DIR.mkdir(exist_ok=True)

# ── Sandboxed execution environment ────────────────────────────────

# Pre-import all diagrams modules so the LLM code doesn't need imports
_DIAGRAM_GLOBALS: dict[str, Any] = {}


def _init_diagram_globals():
    """Build the execution namespace with all diagrams imports pre-loaded."""
    global _DIAGRAM_GLOBALS
    if _DIAGRAM_GLOBALS:
        return

    import diagrams
    import diagrams.aws.analytics
    import diagrams.aws.compute
    import diagrams.aws.database
    import diagrams.aws.integration
    import diagrams.aws.management
    import diagrams.aws.ml
    import diagrams.aws.network
    import diagrams.aws.security
    import diagrams.aws.storage
    import diagrams.aws.general
    import diagrams.aws.devtools
    import diagrams.aws.game
    import diagrams.aws.iot
    import diagrams.aws.media
    import diagrams.aws.migration
    import diagrams.aws.mobile
    import diagrams.aws.engagement
    import diagrams.aws.business
    import diagrams.aws.cost
    import diagrams.aws.enduser
    import diagrams.aws.robotics
    import diagrams.aws.quantum
    import diagrams.aws.satellite
    import diagrams.aws.blockchain

    # Also import common non-AWS providers
    try:
        import diagrams.onprem.compute
        import diagrams.onprem.database
        import diagrams.onprem.network
        import diagrams.onprem.client
        import diagrams.onprem.container
        import diagrams.generic.compute
        import diagrams.generic.database
        import diagrams.generic.network
        import diagrams.generic.storage
    except ImportError:
        pass

    # Build namespace: import everything into a flat dict
    ns: dict[str, Any] = {"__builtins__": __builtins__}

    # Core diagrams classes
    ns["Diagram"] = diagrams.Diagram
    ns["Cluster"] = diagrams.Cluster
    ns["Edge"] = diagrams.Edge
    ns["Node"] = diagrams.Node

    # Import all AWS modules by name
    import pkgutil
    for _, mod_name, _ in pkgutil.iter_modules(diagrams.aws.__path__):
        try:
            mod = __import__(f"diagrams.aws.{mod_name}", fromlist=[mod_name])
            ns[f"aws_{mod_name}"] = mod
            # Also expose all classes directly for convenience
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, diagrams.Node) and obj is not diagrams.Node:
                    ns[attr] = obj
        except ImportError:
            pass

    # Import onprem/generic if available
    for provider in ("onprem", "generic", "k8s", "saas"):
        try:
            provider_mod = __import__(f"diagrams.{provider}", fromlist=[provider])
            for _, mod_name, _ in pkgutil.iter_modules(provider_mod.__path__):
                try:
                    mod = __import__(f"diagrams.{provider}.{mod_name}", fromlist=[mod_name])
                    ns[f"{provider}_{mod_name}"] = mod
                    for attr in dir(mod):
                        obj = getattr(mod, attr)
                        if isinstance(obj, type) and issubclass(obj, diagrams.Node) and obj is not diagrams.Node:
                            # Prefix non-AWS classes to avoid collision
                            if attr not in ns:
                                ns[attr] = obj
                except ImportError:
                    pass
        except ImportError:
            pass

    # Custom class for user-supplied icons
    try:
        from diagrams.custom import Custom
        ns["Custom"] = Custom
    except ImportError:
        pass

    _DIAGRAM_GLOBALS = ns


# ── Tool: generate_diagram ──────────────────────────────────────────

@tool
def generate_diagram(code: str, filename: str = "") -> dict:
    """Generate an AWS architecture diagram from Python code using the diagrams package.

    The code should use the `diagrams` package DSL. All imports are pre-loaded —
    start directly with `with Diagram(...)`. Available classes include all AWS
    service icons (e.g., EC2, Lambda, RDS, S3, ELB, CloudFront, DynamoDB, etc.)
    plus Diagram, Cluster, Edge, and Node.

    CODE REQUIREMENTS:
    - Must include a `with Diagram(...)` block
    - Do NOT include any import statements — everything is pre-imported
    - Use `show=False` in the Diagram constructor (this is enforced automatically)
    - Use `direction="LR"` for left-to-right flow (recommended)

    COMMON PATTERNS:
    - Basic: `ELB("lb") >> EC2("web") >> RDS("db")`
    - Fan-out: `ELB("lb") >> [EC2("web1"), EC2("web2")] >> RDS("db")`
    - Grouping: `with Cluster("VPC"): ...`
    - Styling: `svc1 >> Edge(color="red", style="dashed") >> svc2`

    Args:
        code: Python code using the diagrams package DSL. No imports needed.
        filename: Optional filename (without extension). Auto-generated if empty.

    Returns:
        Dictionary with 'path' to the generated PNG and 'status'.
    """
    _init_diagram_globals()

    if not filename:
        filename = f"diagram_{uuid.uuid4().hex[:8]}"

    # Sanitize filename
    filename = re.sub(r'[^\w\-]', '_', filename).strip('_')
    output_path = str(DIAGRAM_DIR / filename)

    # Patch the code to enforce show=False and set the output path
    # Replace any Diagram(...) call to inject our filename and show=False
    patched_code = code

    # Remove any import lines (agent shouldn't need them but just in case)
    patched_code = re.sub(r'^(from|import)\s+.*$', '', patched_code, flags=re.MULTILINE)

    # Inject filename and show=False into Diagram() constructor
    # Match: with Diagram( or Diagram(
    def _patch_diagram_call(match):
        prefix = match.group(0)
        # We'll add our params after the opening paren
        return prefix

    # Strategy: set outformat and filename via environment-like injection
    # We prepend variable assignments the code can reference
    exec_code = f'_output_path = "{output_path}"\n' + patched_code

    # Ensure show=False is present — replace show=True if found
    exec_code = re.sub(r'\bshow\s*=\s*True', 'show=False', exec_code)

    # If no show= parameter at all, add it after Diagram(
    if 'show=' not in exec_code and 'show =' not in exec_code:
        exec_code = re.sub(
            r'(with\s+Diagram\s*\([^)]*)',
            r'\1, show=False',
            exec_code
        )

    # Inject filename= if not present
    if 'filename=' not in exec_code and 'filename =' not in exec_code:
        exec_code = re.sub(
            r'(with\s+Diagram\s*\([^)]*)',
            f'\\1, filename=_output_path',
            exec_code
        )
    else:
        # Override any existing filename with our path
        exec_code = re.sub(
            r'filename\s*=\s*["\'][^"\']*["\']',
            f'filename=_output_path',
            exec_code
        )

    # Execute in sandboxed namespace
    exec_ns = dict(_DIAGRAM_GLOBALS)
    exec_ns["_output_path"] = output_path

    try:
        exec(exec_code, exec_ns)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("Diagram generation failed:\n%s\n\nCode:\n%s", error_msg, exec_code[:500])
        return {
            "status": "error",
            "error": error_msg,
            "traceback": traceback.format_exc()[-500:],
        }

    # Check if file was created
    png_path = f"{output_path}.png"
    if os.path.exists(png_path):
        return {
            "status": "success",
            "path": png_path,
            "filename": f"{filename}.png",
            "message": f"Diagram generated: {filename}.png",
        }
    else:
        # Sometimes diagrams saves without the .png extension
        if os.path.exists(output_path):
            os.rename(output_path, png_path)
            return {
                "status": "success",
                "path": png_path,
                "filename": f"{filename}.png",
                "message": f"Diagram generated: {filename}.png",
            }
        return {
            "status": "error",
            "error": f"Diagram code executed but no output file found at {png_path}",
        }


# ── Tool: get_diagram_examples ──────────────────────────────────────

@tool
def get_diagram_examples(diagram_type: str = "aws") -> dict:
    """Get example code for different types of AWS architecture diagrams.

    Use these examples to understand the syntax before generating your own.
    Remember: NO import statements needed — all classes are pre-imported.

    Args:
        diagram_type: Type of example. Options: "aws", "serverless", "microservices", "data", "all"

    Returns:
        Dictionary with example code snippets for the requested type.
    """
    examples = {
        "aws": {
            "basic_web_app": '''with Diagram("Web Application", filename=_output_path, show=False, direction="LR"):
    dns = Route53("DNS")
    lb = ELB("ALB")
    with Cluster("Auto Scaling Group"):
        web = [EC2("web1"), EC2("web2"), EC2("web3")]
    db = RDS("PostgreSQL")
    cache = ElastiCache("Redis")

    dns >> lb >> web >> db
    web >> cache''',

            "three_tier": '''with Diagram("Three-Tier Architecture", filename=_output_path, show=False, direction="LR"):
    users = Users("Users")

    with Cluster("Presentation Tier"):
        cf = CloudFront("CDN")
        s3 = S3("Static Assets")

    with Cluster("Application Tier"):
        lb = ELB("ALB")
        with Cluster("ECS Cluster"):
            svcs = [ECS("svc1"), ECS("svc2")]

    with Cluster("Data Tier"):
        db = RDS("Aurora")
        dynamo = DynamoDB("Sessions")

    users >> cf >> lb >> svcs >> db
    svcs >> dynamo
    cf >> s3''',
        },

        "serverless": {
            "api_backend": '''with Diagram("Serverless API", filename=_output_path, show=False, direction="LR"):
    client = Users("Client")
    apigw = APIGateway("API Gateway")

    with Cluster("Lambda Functions"):
        create = Lambda("create")
        read = Lambda("read")
        update = Lambda("update")
        delete = Lambda("delete")

    db = DynamoDB("DynamoDB")
    queue = SQS("Async Queue")
    notify = SNS("Notifications")

    client >> apigw >> [create, read, update, delete] >> db
    create >> queue >> Lambda("processor") >> notify''',
        },

        "microservices": {
            "event_driven": '''with Diagram("Event-Driven Microservices", filename=_output_path, show=False, direction="LR"):
    with Cluster("API Layer"):
        apigw = APIGateway("Gateway")

    with Cluster("Services"):
        orders = ECS("Orders")
        payments = ECS("Payments")
        inventory = ECS("Inventory")
        shipping = ECS("Shipping")

    with Cluster("Event Bus"):
        bus = Eventbridge("EventBridge")

    with Cluster("Data Stores"):
        orders_db = DynamoDB("Orders DB")
        inv_db = RDS("Inventory DB")

    apigw >> orders >> bus
    bus >> [payments, inventory, shipping]
    orders >> orders_db
    inventory >> inv_db''',
        },

        "data": {
            "analytics_pipeline": '''with Diagram("Data Analytics Pipeline", filename=_output_path, show=False, direction="LR"):
    sources = [S3("Raw Data"), Kinesis("Stream")]

    with Cluster("Processing"):
        glue = Glue("ETL Jobs")
        emr = EMR("Spark")

    with Cluster("Storage"):
        lake = S3("Data Lake")
        warehouse = Redshift("Warehouse")

    with Cluster("Analytics"):
        athena = Athena("Athena")
        quicksight = Quicksight("Dashboard")

    sources >> glue >> lake
    lake >> emr >> warehouse
    warehouse >> athena >> quicksight''',
        },
    }

    if diagram_type == "all":
        return {"examples": examples}
    elif diagram_type in examples:
        return {"examples": {diagram_type: examples[diagram_type]}}
    else:
        return {"examples": examples.get("aws", {}),
                "available_types": list(examples.keys())}


# ── Tool: list_diagram_icons ────────────────────────────────────────

@tool
def list_diagram_icons(provider: str = "aws", service_filter: str = "") -> dict:
    """List available diagram icons for a provider, with optional service filtering.

    Use this to discover which icon class names are available before writing diagram code.

    Args:
        provider: Icon provider. Options: "aws", "onprem", "generic", "k8s", "saas"
        service_filter: Optional filter for a specific service category (e.g., "compute", "database", "network")

    Returns:
        Dictionary mapping service categories to lists of available icon class names.
    """
    import pkgutil
    import diagrams

    result: dict[str, list[str]] = {}

    try:
        provider_mod = __import__(f"diagrams.{provider}", fromlist=[provider])
    except ImportError:
        return {"error": f"Provider '{provider}' not found. Available: aws, onprem, generic, k8s, saas"}

    for _, mod_name, _ in pkgutil.iter_modules(provider_mod.__path__):
        if service_filter and service_filter.lower() not in mod_name.lower():
            continue
        try:
            mod = __import__(f"diagrams.{provider}.{mod_name}", fromlist=[mod_name])
            icons = [
                attr for attr in dir(mod)
                if not attr.startswith('_')
                and isinstance(getattr(mod, attr, None), type)
                and issubclass(getattr(mod, attr), diagrams.Node)
                and getattr(mod, attr) is not diagrams.Node
            ]
            if icons:
                result[mod_name] = sorted(icons)
        except ImportError:
            pass

    return {"provider": provider, "services": result}
