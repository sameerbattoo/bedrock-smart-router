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

## Guardrails Integration (Phase 3, optional)

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

## Least-Privilege Recommendation

For production, combine only the statements you need. A typical Lambda running the router with DynamoDB metrics and no auto-create:

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
            "Sid": "MetricsStore",
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            "Resource": "arn:aws:dynamodb:us-west-2:123456789012:table/BedrockSmartRouterMetrics"
        }
    ]
}
```
