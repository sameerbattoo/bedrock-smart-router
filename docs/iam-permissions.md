# IAM Permissions Reference

This document lists the IAM permissions required by the Bedrock Smart Router depending on which features you enable.

---

## Bedrock Inference (always required)

The router calls `bedrock-runtime:Converse` and `bedrock-runtime:ConverseStream` to invoke models.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockInference",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "arn:aws:bedrock:*:*:inference-profile/*"
        }
    ]
}
```

To restrict to specific models, replace the wildcard resource with specific model ARNs:

```
arn:aws:bedrock:us-west-2::foundation-model/us.anthropic.claude-sonnet-4-6
arn:aws:bedrock:us-west-2::foundation-model/us.amazon.nova-pro-v1:0
```

### Global CRIS Profiles

If the router selects global CRIS profiles (`global.*`), the IAM policy needs three parts — the regional inference profile, the regional foundation model, and the global foundation model. See the [AWS documentation on global cross-region inference IAM](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html) for the full policy structure. The simplest approach is to use the wildcard resource above, which covers all profiles.

---

## DynamoDB Metrics Store

Required only when `metrics.backend` is set to `"dynamodb"`.

### Option A: auto_create_table=True (default, recommended for dev/test)

The router creates the table on first use and enables TTL.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockRouterMetricsDynamoDB",
            "Effect": "Allow",
            "Action": [
                "dynamodb:CreateTable",
                "dynamodb:DescribeTable",
                "dynamodb:UpdateTimeToLive",
                "dynamodb:ListTables",
                "dynamodb:PutItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            "Resource": "arn:aws:dynamodb:*:*:table/BedrockSmartRouterMetrics"
        }
    ]
}
```

### Option B: auto_create_table=False (recommended for production)

You create the table yourself. The router only reads and writes data.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockRouterMetricsDynamoDB",
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            "Resource": "arn:aws:dynamodb:*:*:table/BedrockSmartRouterMetrics"
        }
    ]
}
```

### Pre-provisioning the table

**CloudFormation:**

```yaml
BedrockRouterMetricsTable:
  Type: AWS::DynamoDB::Table
  Properties:
    TableName: BedrockSmartRouterMetrics
    BillingMode: PAY_PER_REQUEST
    KeySchema:
      - AttributeName: model_id
        KeyType: HASH
      - AttributeName: timestamp
        KeyType: RANGE
    AttributeDefinitions:
      - AttributeName: model_id
        AttributeType: S
      - AttributeName: timestamp
        AttributeType: N
    TimeToLiveSpecification:
      AttributeName: expires_at
      Enabled: true
```

**CDK (Python):**

```python
from aws_cdk import aws_dynamodb as dynamodb, RemovalPolicy

table = dynamodb.Table(
    self, "BedrockRouterMetrics",
    table_name="BedrockSmartRouterMetrics",
    partition_key=dynamodb.Attribute(
        name="model_id", type=dynamodb.AttributeType.STRING
    ),
    sort_key=dynamodb.Attribute(
        name="timestamp", type=dynamodb.AttributeType.NUMBER
    ),
    billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
    time_to_live_attribute="expires_at",
    removal_policy=RemovalPolicy.RETAIN,
)
```

**Terraform:**

```hcl
resource "aws_dynamodb_table" "bedrock_router_metrics" {
  name         = "BedrockSmartRouterMetrics"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "model_id"
  range_key    = "timestamp"

  attribute {
    name = "model_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}
```

Replace `BedrockSmartRouterMetrics` with your custom `table_name` if you changed it in the router config.

---

## Pricing Refresh (optional)

Required only if you call `PricingRefresher.refresh_from_bedrock()` or `refresh_from_pricing_api()`.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockModelDiscovery",
            "Effect": "Allow",
            "Action": ["bedrock:ListFoundationModels"],
            "Resource": "*"
        },
        {
            "Sid": "PricingAPI",
            "Effect": "Allow",
            "Action": ["pricing:GetProducts"],
            "Resource": "*"
        }
    ]
}
```

---

## CloudWatch Metrics (optional)

Required only when `observability.cloudwatch_enabled` is set to `true`.

The router publishes custom metrics (RoutingDecisions, Latency, Cost, CacheHits, FallbacksUsed, CircuitBreakerSkips, CostSavings) to a configurable CloudWatch namespace.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockRouterCloudWatchMetrics",
            "Effect": "Allow",
            "Action": ["cloudwatch:PutMetricData"],
            "Resource": "*"
        }
    ]
}
```

Note: `PutMetricData` does not support resource-level restrictions — the `Resource` must be `"*"`. The namespace is controlled in the router config (`observability.cloudwatch_namespace`, default `"BedrockSmartRouter"`).

If you also want to query metrics (e.g. for dashboards or the integration test), add:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockRouterCloudWatchRead",
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "cloudwatch:GetMetricData",
                "cloudwatch:ListMetrics"
            ],
            "Resource": "*"
        }
    ]
}
```

**Configuration:**

```yaml
observability:
  log_decisions: true
  cloudwatch_enabled: true
  cloudwatch_namespace: "MyApp/BedrockRouter"
```

**Published metrics:**

| Metric | Unit | Dimensions | When |
|---|---|---|---|
| `RoutingDecisions` | Count | Model, Strategy, Complexity | Every request |
| `Latency` | Milliseconds | Model, Strategy, Complexity | When latency > 0 |
| `Cost` | None (USD) | Model, Strategy, Complexity | When cost > 0 |
| `CacheHits` | Count | Model, Strategy, Complexity | On cache hits |
| `FallbacksUsed` | Count | Model, Strategy, Complexity | When fallback triggered |
| `CircuitBreakerSkips` | Count | Model, Strategy, Complexity | When models skipped |
| `CostSavings` | None (USD) | Model, Strategy, Complexity | When routing saved money |

---

## Guardrails Integration (optional)

Required only when pre-route or post-route guardrails are configured.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockGuardrails",
            "Effect": "Allow",
            "Action": ["bedrock:ApplyGuardrail"],
            "Resource": "arn:aws:bedrock:*:*:guardrail/*"
        }
    ]
}
```

---

## Application Inference Profiles / Multi-Tenant (optional)

Required only when `aip.enabled` is `true` and `aip.auto_create` is `true`. The router creates per-tenant inference profiles for cost attribution and needs STS to discover the account ID.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockAIPManagement",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateInferenceProfile"
            ],
            "Resource": "arn:aws:bedrock:*:*:inference-profile/*"
        },
        {
            "Sid": "STSIdentity",
            "Effect": "Allow",
            "Action": ["sts:GetCallerIdentity"],
            "Resource": "*"
        }
    ]
}
```

If `auto_create` is `false`, you create the inference profiles yourself and the router only needs `bedrock:InvokeModel` (already covered above).

---

## Provisioned Throughput Detection (optional)

Required only when `provisioned_throughput.enabled` is `true`. The router lists active provisioned capacity to prefer already-paid throughput.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockProvisionedThroughput",
            "Effect": "Allow",
            "Action": ["bedrock:ListProvisionedModelThroughputs"],
            "Resource": "*"
        }
    ]
}
```

---

## Semantic Cache & Semantic Router (optional)

Required only when using `SemanticCache` or `SemanticRouter`. These call the Bedrock embedding model (default: `amazon.titan-embed-text-v2:0`) via `InvokeModel`.

The Bedrock Inference permission above covers this if you use a wildcard resource. If you restrict to specific model ARNs, add the embedding model:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockEmbeddings",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel"],
            "Resource": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"
        }
    ]
}
```

---

## Redis / Valkey / ElastiCache (optional)

Redis and Valkey are accessed over the network (TCP/TLS), not via AWS APIs. No IAM permissions are needed — authentication is handled by the Redis connection URL (password or IAM auth token).

For ElastiCache with IAM authentication, the application needs network access to the VPC (security group rules) and the ElastiCache IAM auth token. See [ElastiCache IAM authentication](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/auth-iam.html).

---

## Least-Privilege Recommendation

For production, combine only the statements you need. A typical Lambda running the router with DynamoDB metrics, CloudWatch observability, guardrails, and no auto-create:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockInference",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "arn:aws:bedrock:us-west-2:123456789012:inference-profile/*"
        },
        {
            "Sid": "BedrockGuardrails",
            "Effect": "Allow",
            "Action": ["bedrock:ApplyGuardrail"],
            "Resource": "arn:aws:bedrock:us-west-2:123456789012:guardrail/*"
        },
        {
            "Sid": "MetricsStore",
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            "Resource": "arn:aws:dynamodb:us-west-2:123456789012:table/BedrockSmartRouterMetrics"
        },
        {
            "Sid": "CloudWatchMetrics",
            "Effect": "Allow",
            "Action": ["cloudwatch:PutMetricData"],
            "Resource": "*"
        }
    ]
}
```

Add these only if you use the corresponding features:

| Feature | Add these actions |
|---|---|
| Multi-tenant AIPs (`aip.auto_create: true`) | `bedrock:CreateInferenceProfile`, `sts:GetCallerIdentity` |
| Provisioned throughput detection | `bedrock:ListProvisionedModelThroughputs` |
| Semantic cache / semantic router | `bedrock:InvokeModel` on the embedding model ARN |
| Pricing refresh | `bedrock:ListFoundationModels`, `pricing:GetProducts` |
