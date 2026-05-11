# Benchmark Analysis Report

**Generated:** 2026-05-09 13:15:32
**Total results:** 150
**Successful:** 150
**Failed:** 0
**Runners:** sonnet, haiku, nova-pro, router-default, router-quality

**Region:** us-west-2
**Judge model:** us.anthropic.claude-sonnet-4-6

## Overall Summary

| Runner | Count | Avg Latency | Avg Score | Avg Cost | Success Rate |
|--------|-------|-------------|-----------|----------|--------------|
| Claude Sonnet 4.6 | 30 | 5090ms | 8.73/10 | $0.005974 | 100.0% |
| Claude Haiku 4.5 | 30 | 2076ms | 8.50/10 | $0.001251 | 100.0% |
| Amazon Nova Pro | 30 | 1333ms | 7.80/10 | $0.001003 | 100.0% |
| Smart Router (Default/Balanced) | 30 | 1129ms | 7.03/10 | $0.000076 | 100.0% |
| Smart Router (Quality) | 30 | 4349ms | 8.63/10 | $0.011917 | 100.0% |

## Quality Score by Category

| Category | Claude Sonne | Claude Haiku | Amazon Nova  | Smart Router | Smart Router |
|----------|------|------|------|------|------|
| anomaly_detection | 8.6 | 8.0 | 5.6 | 4.8 | 8.6 |
| code_generation | 9.2 | 9.4 | 8.8 | 9.0 | 8.8 |
| document_extraction | 8.2 | 8.8 | 6.8 | 5.8 | 8.2 |
| log_analysis | 8.6 | 7.6 | 7.6 | 4.6 | 7.8 |
| summarization | 8.2 | 7.8 | 8.4 | 8.4 | 8.6 |
| text_to_sql | 9.6 | 9.4 | 9.6 | 9.6 | 9.8 |

## Quality Score by Difficulty

| Difficulty | Claude Sonne | Claude Haiku | Amazon Nova  | Smart Router | Smart Router |
|------------|------|------|------|------|------|
| simple | 8.7 | 8.5 | 7.8 | 7.0 | 8.6 |
| medium | - | - | - | - | - |
| complex | - | - | - | - | - |

## Cost Analysis

| Runner | Total Cost | Avg/Prompt | vs Sonnet Savings |
|--------|-----------|-----------|-------------------|
| Claude Sonnet 4.6 | $0.1792 | $0.005974 | baseline |
| Claude Haiku 4.5 | $0.0375 | $0.001251 | +79.1% |
| Amazon Nova Pro | $0.0301 | $0.001003 | +83.2% |
| Smart Router (Default/Balanced) | $0.0023 | $0.000076 | +98.7% |
| Smart Router (Quality) | $0.3575 | $0.011917 | -99.5% |

## Latency Analysis

| Runner | Avg | Min | Max | p50 | p95 |
|--------|-----|-----|-----|-----|-----|
| Claude Sonnet 4.6 | 5090ms | 837ms | 23403ms | 2656ms | 16075ms |
| Claude Haiku 4.5 | 2076ms | 604ms | 6382ms | 1571ms | 5243ms |
| Amazon Nova Pro | 1333ms | 350ms | 4374ms | 910ms | 4115ms |
| Smart Router (Default/Balanced) | 1129ms | 309ms | 4175ms | 761ms | 3830ms |
| Smart Router (Quality) | 4349ms | 1045ms | 13166ms | 3291ms | 12703ms |

## Router Model Selection Distribution

### Smart Router (Default/Balanced)

| Model | Count | Percentage |
|-------|-------|------------|
| us.amazon.nova-lite-v1:0 | 25 | 83.3% |
| us.amazon.nova-micro-v1:0 | 5 | 16.7% |

### Smart Router (Quality)

| Model | Count | Percentage |
|-------|-------|------------|
| global.anthropic.claude-opus-4-7 | 30 | 100.0% |

## Key Insights

- **Best quality:** Claude Sonnet 4.6 (8.73/10)
- **Cheapest:** Smart Router (Default/Balanced) ($0.000076/prompt)
- **Fastest:** Smart Router (Default/Balanced) (1129ms avg)

### Router (Default) vs Sonnet Comparison

| Metric | Router (Default) | Sonnet | Difference |
|--------|-----------------|--------|------------|
| Avg Cost | $0.000076 | $0.005974 | -98.7% |
| Avg Latency | 1129ms | 5090ms | -77.8% |
| Avg Quality | 7.03/10 | 8.73/10 | -1.70 |
