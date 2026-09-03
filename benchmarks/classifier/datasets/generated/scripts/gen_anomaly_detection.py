# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Generate anomaly detection benchmark prompts."""
import json
import os

# Write generated JSON into the parent generated/ dir (alongside the committed datasets)
PROMPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(PROMPTS_DIR, exist_ok=True)

TIMESERIES_DATA = """timestamp,cpu_percent,memory_mb,request_count,error_count,latency_ms,disk_io_mbps
2024-03-01 00:00,45.2,3200,1200,2,85,12.5
2024-03-01 01:00,42.1,3180,980,1,82,11.2
2024-03-01 02:00,38.5,3150,750,0,78,9.8
2024-03-01 03:00,35.8,3120,620,1,75,8.5
2024-03-01 04:00,36.2,3130,680,0,76,9.0
2024-03-01 05:00,40.5,3160,890,1,80,10.5
2024-03-01 06:00,52.3,3280,1500,3,92,15.2
2024-03-01 07:00,65.8,3450,2200,4,105,18.7
2024-03-01 08:00,78.2,3680,3100,5,125,22.3
2024-03-01 09:00,82.5,3750,3500,6,135,24.1
2024-03-01 10:00,85.1,3820,3800,8,142,25.5
2024-03-01 11:00,87.3,3900,4000,7,148,26.8
2024-03-01 12:00,84.6,3850,3700,5,138,25.0
2024-03-01 13:00,86.2,3880,3900,6,145,26.2
2024-03-01 14:00,88.5,3950,4100,9,155,27.5
2024-03-01 15:00,92.8,4200,4500,45,320,35.2
2024-03-01 16:00,95.1,4500,4800,120,850,42.8
2024-03-01 17:00,78.5,3700,3200,8,130,22.0
2024-03-01 18:00,68.2,3500,2500,4,110,18.5
2024-03-01 19:00,58.5,3350,1800,3,95,15.0
2024-03-01 20:00,52.1,3250,1400,2,88,13.2
2024-03-01 21:00,48.3,3200,1100,1,84,12.0
2024-03-01 22:00,45.8,3180,950,1,82,11.5
2024-03-01 23:00,43.2,3160,850,0,80,10.8"""

FINANCIAL_DATA = """date,transaction_id,customer_id,amount,merchant,category,country,time_of_day,device
2024-03-01,TXN001,C100,45.99,Amazon,retail,US,14:23,mobile
2024-03-01,TXN002,C100,12.50,Starbucks,food,US,14:45,mobile
2024-03-01,TXN003,C100,89.99,BestBuy,electronics,US,15:10,desktop
2024-03-01,TXN004,C100,2500.00,WireTransfer,transfer,NG,15:12,desktop
2024-03-01,TXN005,C100,1800.00,CryptoExchange,crypto,RU,15:13,desktop
2024-03-01,TXN006,C100,3200.00,WireTransfer,transfer,NG,15:14,unknown
2024-03-02,TXN007,C201,125.00,Walmart,retail,US,09:30,mobile
2024-03-02,TXN008,C201,67.50,GasStation,fuel,US,10:15,card_present
2024-03-02,TXN009,C201,234.00,HomeDepot,home,US,11:00,card_present
2024-03-02,TXN010,C201,45.00,Restaurant,food,US,12:30,card_present
2024-03-02,TXN011,C201,89.00,Restaurant,food,FR,12:35,card_present
2024-03-02,TXN012,C201,156.00,LuxuryStore,retail,FR,13:00,card_present
2024-03-03,TXN013,C305,15.99,Netflix,subscription,US,20:00,mobile
2024-03-03,TXN014,C305,9.99,Spotify,subscription,US,20:00,mobile
2024-03-03,TXN015,C305,15.99,Netflix,subscription,US,20:01,mobile
2024-03-03,TXN016,C305,15.99,Netflix,subscription,US,20:01,mobile
2024-03-03,TXN017,C305,9.99,Spotify,subscription,US,20:02,mobile
2024-03-03,TXN018,C305,9.99,Spotify,subscription,US,20:02,mobile
2024-03-04,TXN019,C412,5000.00,Payroll,income,US,09:00,desktop
2024-03-04,TXN020,C412,4800.00,WireTransfer,transfer,US,09:05,desktop
2024-03-04,TXN021,C412,150.00,ATM,withdrawal,US,09:30,card_present
2024-03-04,TXN022,C412,150.00,ATM,withdrawal,MX,09:35,card_present
2024-03-04,TXN023,C412,150.00,ATM,withdrawal,MX,09:40,card_present"""

NETWORK_DATA = """timestamp,source_ip,dest_ip,dest_port,protocol,bytes_sent,bytes_recv,duration_ms,flags
2024-03-15 02:00:01,10.0.1.50,10.0.2.100,443,TCP,1200,45000,250,SYN-ACK
2024-03-15 02:00:02,10.0.1.50,10.0.2.100,443,TCP,800,32000,180,ACK
2024-03-15 02:00:05,10.0.1.50,10.0.2.101,443,TCP,1100,41000,220,SYN-ACK
2024-03-15 02:01:00,10.0.1.50,192.168.1.1,53,UDP,64,128,5,
2024-03-15 02:01:01,10.0.1.50,8.8.8.8,53,UDP,64,256,15,
2024-03-15 02:01:02,10.0.1.50,185.199.108.1,443,TCP,5000,120000,1500,SYN-ACK
2024-03-15 02:01:03,10.0.1.50,185.199.108.1,443,TCP,450000,2000,3000,ACK-PSH
2024-03-15 02:02:00,10.0.1.50,10.0.2.100,22,TCP,500,200,50,SYN-ACK
2024-03-15 02:02:01,10.0.1.50,10.0.2.101,22,TCP,500,200,50,SYN-ACK
2024-03-15 02:02:02,10.0.1.50,10.0.2.102,22,TCP,500,200,50,SYN-ACK
2024-03-15 02:02:03,10.0.1.50,10.0.2.103,22,TCP,500,200,50,SYN-ACK
2024-03-15 02:02:04,10.0.1.50,10.0.2.104,22,TCP,500,200,50,SYN-ACK
2024-03-15 02:02:05,10.0.1.50,10.0.2.105,22,TCP,0,0,5000,SYN
2024-03-15 02:02:06,10.0.1.50,10.0.2.106,22,TCP,0,0,5000,SYN
2024-03-15 02:02:07,10.0.1.50,10.0.2.107,22,TCP,0,0,5000,SYN
2024-03-15 02:03:00,10.0.1.50,45.33.32.156,4444,TCP,50000,100,500,SYN-ACK
2024-03-15 02:03:30,10.0.1.50,45.33.32.156,4444,TCP,120000,50,1000,ACK-PSH
2024-03-15 02:04:00,10.0.1.50,45.33.32.156,4444,TCP,250000,100,2000,ACK-PSH"""

METRICS_DATA = """service,timestamp,requests_per_sec,error_rate_pct,p50_ms,p95_ms,p99_ms,cpu_pct,memory_pct,active_connections
api-gateway,2024-03-15 10:00,500,0.2,15,45,120,35,42,200
api-gateway,2024-03-15 10:05,520,0.3,16,48,125,37,43,210
api-gateway,2024-03-15 10:10,510,0.2,15,46,118,36,42,205
api-gateway,2024-03-15 10:15,530,0.4,17,50,130,38,44,215
api-gateway,2024-03-15 10:20,1200,2.5,45,250,800,72,68,480
api-gateway,2024-03-15 10:25,1500,8.5,120,500,2000,89,82,620
api-gateway,2024-03-15 10:30,800,15.2,250,1200,5000,95,91,750
api-gateway,2024-03-15 10:35,400,25.0,500,2500,8000,98,95,900
api-gateway,2024-03-15 10:40,200,45.0,1000,5000,15000,99,98,950
api-gateway,2024-03-15 10:45,150,60.0,2000,8000,20000,99,99,980
user-service,2024-03-15 10:00,200,0.1,25,80,200,28,35,100
user-service,2024-03-15 10:05,210,0.1,26,82,205,29,36,105
user-service,2024-03-15 10:10,205,0.2,25,81,198,28,35,102
user-service,2024-03-15 10:15,215,0.1,27,83,210,30,37,108
user-service,2024-03-15 10:20,450,1.2,55,200,600,58,55,250
user-service,2024-03-15 10:25,600,5.0,100,400,1500,75,72,380
user-service,2024-03-15 10:30,300,12.0,200,800,3000,82,80,450
payment-service,2024-03-15 10:00,100,0.0,50,150,300,22,30,50
payment-service,2024-03-15 10:05,105,0.0,52,155,310,23,31,52
payment-service,2024-03-15 10:10,98,0.0,49,148,295,22,30,49
payment-service,2024-03-15 10:15,102,0.0,51,152,305,22,30,51
payment-service,2024-03-15 10:20,100,0.0,50,150,300,22,30,50
payment-service,2024-03-15 10:25,95,0.5,55,160,320,24,32,55
payment-service,2024-03-15 10:30,80,2.0,65,180,400,26,34,60"""

prompts = [
    # ==================== SIMPLE (17 prompts) ====================
    {"id": "anom_001", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "Identify the time period with anomalous CPU and error count spikes.",
     "context": TIMESERIES_DATA,
     "expected_answer": "Anomaly detected at 15:00-16:00. CPU spiked from normal range (35-88%) to 92.8-95.1%. Error count jumped from normal (0-9) to 45 at 15:00 and 120 at 16:00. Latency increased from ~155ms to 320ms then 850ms. This correlates with increased request count (4500-4800) suggesting a traffic spike or DDoS that overwhelmed the system."},
    {"id": "anom_002", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a fraud detection analyst. Analyze the transaction data and flag suspicious activity.",
     "user_prompt": "Which transactions for customer C100 appear fraudulent and why?",
     "context": FINANCIAL_DATA,
     "expected_answer": "TXN004, TXN005, TXN006 are suspicious. Reasons: 1) Sudden large amounts ($2500, $1800, $3200) after small purchases ($45-$89). 2) Transactions to Nigeria and Russia within 2 minutes of each other. 3) Wire transfers and crypto exchange - high-risk categories. 4) TXN006 from 'unknown' device. 5) All three occurred within 2 minutes (15:12-15:14) suggesting automated fraud after account compromise."},
    {"id": "anom_003", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a network security analyst. Analyze the network traffic data and identify suspicious patterns.",
     "user_prompt": "What suspicious activity is happening from 10.0.1.50?",
     "context": NETWORK_DATA,
     "expected_answer": "Multiple suspicious activities: 1) SSH scanning (02:02:00-02:02:07) - rapid connections to sequential IPs on port 22, some with SYN-only (no response = port scan). 2) Data exfiltration (02:01:02-02:01:03) - large upload (450KB) to external IP 185.199.108.1. 3) C2 communication (02:03:00-02:04:00) - connections to 45.33.32.156 on port 4444 (common reverse shell port) with large outbound data (420KB total). This pattern suggests a compromised host performing lateral movement and data exfiltration."},
    {"id": "anom_004", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a fraud detection analyst. Analyze the transaction data and flag suspicious activity.",
     "user_prompt": "Are there any duplicate or repeated transactions that look like a system glitch?",
     "context": FINANCIAL_DATA,
     "expected_answer": "Customer C305 has duplicate transactions: Netflix charged 3 times ($15.99 each at 20:00 and 20:01) and Spotify charged 3 times ($9.99 each at 20:00-20:02). Normal subscription billing would be once per month. This looks like a payment processing glitch or retry storm rather than fraud."},
    {"id": "anom_005", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "Which service is experiencing a cascading failure based on the metrics?",
     "context": METRICS_DATA,
     "expected_answer": "The api-gateway is experiencing a cascading failure starting at 10:20. Requests doubled (500->1200), error rate climbed from 0.4% to 60% over 25 minutes, latency went from 17ms p50 to 2000ms, and CPU/memory hit 99%. The user-service is also affected (likely downstream dependency) but payment-service remains relatively stable, suggesting the issue is in the api-gateway or user-service path."},
    {"id": "anom_006", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a fraud detection analyst. Analyze the transaction data and flag suspicious activity.",
     "user_prompt": "Is there anything suspicious about customer C201's transactions?",
     "context": FINANCIAL_DATA,
     "expected_answer": "Moderate suspicion: C201 made purchases in the US (09:30-12:30) then in France (12:35-13:00). The 5-minute gap between a US restaurant and a French restaurant is physically impossible (impossible travel). However, the French transactions are card_present, which could indicate a cloned card being used in France while the real cardholder is in the US."},
    {"id": "anom_007", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "What is the normal baseline for latency and when does it deviate significantly?",
     "context": TIMESERIES_DATA,
     "expected_answer": "Normal baseline latency: 75-155ms (correlates with traffic patterns - lower at night, higher during business hours). Significant deviation: 320ms at 15:00 (2x normal peak) and 850ms at 16:00 (5.5x normal peak). The deviation coincides with the error spike and CPU saturation, indicating system overload."},
    {"id": "anom_008", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a fraud detection analyst. Analyze the transaction data and flag suspicious activity.",
     "user_prompt": "What's suspicious about customer C412's activity?",
     "context": FINANCIAL_DATA,
     "expected_answer": "C412 received $5000 payroll then immediately transferred $4800 (96% of income) via wire transfer. Then made ATM withdrawals in both US and Mexico within 5 minutes (09:30 US, 09:35 MX) - impossible travel. This suggests either account takeover (attacker draining the account after payroll deposit) or money laundering (structuring withdrawals across borders)."},
    {"id": "anom_009", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a network security analyst. Analyze the network traffic data and identify suspicious patterns.",
     "user_prompt": "What does the traffic to port 4444 indicate?",
     "context": NETWORK_DATA,
     "expected_answer": "Port 4444 is commonly used by Metasploit reverse shells and other attack tools. The traffic shows: 1) Connection established to external IP 45.33.32.156. 2) Large outbound data (50KB, then 120KB, then 250KB) with minimal inbound (100 bytes each). 3) This pattern indicates data exfiltration through a reverse shell - the compromised host is sending data out to an attacker-controlled server."},
    {"id": "anom_010", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "Is the payment-service affected by the incident visible in the metrics?",
     "context": METRICS_DATA,
     "expected_answer": "The payment-service is minimally affected. While api-gateway and user-service show severe degradation, payment-service metrics remain mostly stable: requests slightly decreased (100->80), error rate only rose to 2%, and latency increased modestly (50ms->65ms p50). This suggests payment-service is isolated from the cascading failure, possibly because it has its own connection pool or is behind a circuit breaker."},
    {"id": "anom_011", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "What's the correlation between disk I/O and the anomaly period?",
     "context": TIMESERIES_DATA,
     "expected_answer": "Disk I/O follows the same anomaly pattern: normal range is 8.5-27.5 MB/s (correlating with traffic). During the anomaly at 15:00-16:00, disk I/O spiked to 35.2 and 42.8 MB/s respectively. This 55-67% increase above normal peak suggests excessive logging, swap usage due to memory pressure, or a disk-intensive operation (like a runaway query writing temp files)."},
    {"id": "anom_012", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a network security analyst. Analyze the network traffic data and identify suspicious patterns.",
     "user_prompt": "Differentiate between the legitimate and suspicious SSH connections.",
     "context": NETWORK_DATA,
     "expected_answer": "Legitimate SSH (02:02:00-02:02:04): Connections to 10.0.2.100-104 succeeded (500 bytes sent/200 received, SYN-ACK = connection established, 50ms duration). These are likely automated management tasks to known servers. Suspicious SSH (02:02:05-02:02:07): Connections to 10.0.2.105-107 show SYN-only with 5000ms timeout and 0 bytes transferred - these hosts either don't exist or refused the connection. This is port scanning to discover additional targets for lateral movement."},
    {"id": "anom_013", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "What does the active_connections metric tell us about the api-gateway failure?",
     "context": METRICS_DATA,
     "expected_answer": "Active connections grew from 200 (normal) to 980 (4.9x) over 25 minutes while throughput actually decreased (500->150 req/s). This indicates connection exhaustion - requests are piling up but not completing. The system is accepting connections faster than it can process them, leading to thread/connection pool saturation. This is a classic symptom of a downstream dependency timeout causing connection buildup."},
    {"id": "anom_014", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a fraud detection analyst. Analyze the transaction data and flag suspicious activity.",
     "user_prompt": "Rank all customers by fraud risk level (high/medium/low) with brief justification.",
     "context": FINANCIAL_DATA,
     "expected_answer": "HIGH RISK: C100 - Large wire transfers to Nigeria/Russia immediately after normal purchases, unknown device, rapid succession. C412 - Immediate drain of payroll deposit, impossible travel (US+Mexico ATM in 5 min). MEDIUM RISK: C201 - Impossible travel between US and France (5 min gap), but card_present in both locations suggests cloned card. C305 - Duplicate subscription charges, likely system glitch not fraud. LOW RISK: None of the customers show completely normal patterns in this dataset."},
    {"id": "anom_015", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "At what time did the system recover from the anomaly and what evidence supports this?",
     "context": TIMESERIES_DATA,
     "expected_answer": "Recovery occurred at 17:00. Evidence: CPU dropped from 95.1% to 78.5% (back to normal business-hours range), errors dropped from 120 to 8 (normal level), latency returned to 130ms (within normal range), and disk I/O normalized to 22.0 MB/s. The recovery was relatively sharp (one hour), suggesting either the traffic spike ended or an intervention (scaling, blocking, or rate limiting) was applied."},
    {"id": "anom_016", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a network security analyst. Analyze the network traffic data and identify suspicious patterns.",
     "user_prompt": "What is the total volume of data potentially exfiltrated?",
     "context": NETWORK_DATA,
     "expected_answer": "Potential exfiltration: 1) To 185.199.108.1: 450,000 bytes sent (450KB) at 02:01:03. 2) To 45.33.32.156 (C2 server): 50,000 + 120,000 + 250,000 = 420,000 bytes (420KB). Total potential exfiltration: ~870KB. While this seems small, it could contain credentials, database dumps, or compressed sensitive files. The 450KB upload to the external IP is particularly concerning as it's 375x larger than the initial request (1200 bytes)."},
    {"id": "anom_017", "category": "anomaly_detection", "difficulty": "simple",
     "system_prompt": "You are a data analyst specializing in anomaly detection. Analyze the provided data and identify anomalies.",
     "user_prompt": "What is the memory usage pattern and is there a memory leak?",
     "context": TIMESERIES_DATA,
     "expected_answer": "Memory ranges from 3120MB (night low) to 3950MB (day peak) under normal conditions, correlating with traffic. During the anomaly: 4200MB at 15:00 and 4500MB at 16:00 - exceeding normal peak by 300-550MB. After recovery (17:00), memory dropped to 3700MB. Since memory returned to normal after the incident, this is NOT a memory leak - it's traffic-driven memory pressure. A leak would show continuously increasing memory that doesn't return to baseline."},

    # ==================== MEDIUM (18 prompts) ====================
    {"id": "anom_018", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Calculate the z-score for the error count at 15:00 and 16:00 relative to the baseline. Is this statistically significant?",
     "context": TIMESERIES_DATA,
     "expected_answer": "Baseline error counts (excluding 15:00-16:00): [2,1,0,1,0,1,3,4,5,6,8,7,5,6,9,8,4,3,2,1,1,0] -> mean=3.5, std=2.7. At 15:00: error=45, z-score=(45-3.5)/2.7=15.4. At 16:00: error=120, z-score=(120-3.5)/2.7=43.1. Both are extremely significant (z>3 is typically anomalous). Even using a generous threshold of z>5, these are clear outliers at 15.4 and 43.1 standard deviations above the mean."},
    {"id": "anom_019", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Correlate the api-gateway degradation with user-service metrics. What is the dependency relationship and propagation delay?",
     "context": METRICS_DATA,
     "expected_answer": "Correlation analysis: api-gateway anomaly starts at 10:20 (requests 1200, errors 2.5%). user-service anomaly starts at 10:20 simultaneously (requests 450, errors 1.2%). Both degrade together suggesting shared trigger rather than one causing the other. However, api-gateway degrades faster (60% error rate by 10:45 vs 12% for user-service by 10:30). The propagation pattern suggests: external traffic spike hits api-gateway first, which then overwhelms user-service as a downstream dependency. Payment-service isolation confirms it's not a shared infrastructure issue."},
    {"id": "anom_020", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a fraud detection analyst with expertise in behavioral analytics. Analyze patterns across multiple customers.",
     "user_prompt": "Build a velocity-based fraud score for each customer. Consider transaction frequency, amount changes, and geographic spread within time windows.",
     "context": FINANCIAL_DATA,
     "expected_answer": "Velocity fraud scores (0-100): C100: 92/100 - 3 high-value transactions in 2 minutes, 2 countries (NG, RU) never seen before, amount jump from $89 to $2500 (28x increase), device change to 'unknown'. C412: 85/100 - $4800 transfer within 5 min of payroll, ATM in 2 countries within 5 min, velocity of withdrawals (3 in 10 min). C201: 65/100 - Impossible travel (US to FR in 5 min), but amounts are moderate and consistent with shopping pattern. C305: 35/100 - Duplicate charges are high frequency but same merchant/amount pattern suggests system error not fraud."},
    {"id": "anom_021", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Using the timeseries data, calculate the rate of change (derivative) for CPU and identify where acceleration (second derivative) indicates the onset of the anomaly.",
     "context": TIMESERIES_DATA,
     "expected_answer": "CPU rate of change (delta per hour): Normal hours show +/- 2-5% per hour. Key transitions: 14:00->15:00: +4.3%/hr (92.8-88.5), 15:00->16:00: +2.3%/hr. But the acceleration (second derivative) tells the real story: from 13:00-14:00 the rate was +2.3, then 14:00-15:00 jumped to +4.3 - acceleration of +2.0. The anomaly onset is best detected between 14:00-15:00 where the acceleration exceeds normal variance. The error count acceleration is even more dramatic: normal delta is 0-3/hr, but 14:00->15:00 shows +36/hr (9 to 45), giving acceleration of +33 vs previous +3."},
    {"id": "anom_022", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a network security analyst specializing in threat hunting. Analyze traffic patterns to build an attack timeline.",
     "user_prompt": "Reconstruct the complete attack timeline from the network data. What are the distinct phases and their objectives?",
     "context": NETWORK_DATA,
     "expected_answer": "Attack timeline with phases: Phase 1 - Reconnaissance (02:00:01-02:00:05): Normal HTTPS connections to internal servers, possibly establishing baseline or testing access. Phase 2 - DNS Resolution (02:01:00-02:01:01): DNS queries to both internal (192.168.1.1) and external (8.8.8.8) resolvers - looking up external targets. Phase 3 - Data Staging/Exfil (02:01:02-02:01:03): Large data transfer (450KB) to external IP via HTTPS - initial exfiltration disguised as web traffic. Phase 4 - Lateral Movement (02:02:00-02:02:07): SSH scanning of internal subnet, 5 successful + 3 failed connections. Phase 5 - C2 Communication (02:03:00-02:04:00): Reverse shell on port 4444, sending 420KB to attacker. Total attack duration: ~4 minutes."},
    {"id": "anom_023", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Determine the saturation point for the api-gateway. At what threshold does the system transition from degraded to failing?",
     "context": METRICS_DATA,
     "expected_answer": "Saturation analysis shows two distinct transitions: 1) Degraded state (10:20): CPU crosses 70%, error rate exceeds 2%, latency doubles. The system is still processing requests (1200 req/s) but quality degrades. 2) Failure state (10:30-10:35): CPU exceeds 95%, error rate passes 15%, and critically - throughput DROPS (800->400->200 req/s) while connections RISE (750->900->950). The tipping point is at ~90% CPU / 500 active connections where the system can no longer accept new work effectively. The key indicator is the inversion: requests decrease while connections increase, meaning the system is saturated and new requests only add to the queue without being processed."},
    {"id": "anom_024", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a fraud detection analyst with expertise in behavioral analytics. Analyze patterns across multiple customers.",
     "user_prompt": "Design a set of fraud detection rules based on the patterns observed. For each rule, specify the threshold and expected false positive rate.",
     "context": FINANCIAL_DATA,
     "expected_answer": "Proposed rules: 1) Velocity Rule: >3 transactions in 5 minutes with total >$5000 -> Flag. Expected FPR: 2% (legitimate bulk purchases exist). 2) Impossible Travel: Same card used in 2 countries within 1 hour -> Flag. Expected FPR: 5% (VPN purchases, family sharing). 3) Amount Spike: Transaction >10x customer's 30-day average -> Flag. Expected FPR: 8% (legitimate large purchases). 4) High-Risk Destination: Wire transfer to high-risk country (NG, RU) + amount >$1000 -> Block. Expected FPR: 1% (few legitimate large transfers to these countries). 5) Duplicate Detection: Same merchant + same amount within 5 minutes -> Soft flag. Expected FPR: 15% (subscription retries are common). Combined scoring: 2+ rules triggered = block, 1 rule = review."},
    {"id": "anom_025", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Calculate the Pearson correlation coefficient between CPU usage and latency during normal operations vs during the anomaly period. What does this tell us?",
     "context": TIMESERIES_DATA,
     "expected_answer": "Normal period (00:00-14:00): CPU ranges 35-88%, latency 75-155ms. Correlation is approximately r=0.97 (strong linear relationship, ~1.7ms per 1% CPU). Anomaly period (15:00-16:00): CPU 92.8-95.1%, latency 320-850ms. The relationship breaks: 2.3% CPU increase yields 530ms latency increase (230ms per 1% CPU). This 135x amplification indicates the system has crossed a non-linear threshold. During normal operations, latency scales linearly with load. Once saturated, small CPU increases cause exponential latency growth due to queuing effects (Little's Law: as utilization approaches 100%, queue length approaches infinity)."},
    {"id": "anom_026", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a network security analyst specializing in threat hunting. Analyze traffic patterns to build an attack timeline.",
     "user_prompt": "What indicators of compromise (IOCs) can be extracted from this network data for threat intelligence sharing?",
     "context": NETWORK_DATA,
     "expected_answer": "Extractable IOCs: 1) IP Addresses: 45.33.32.156 (C2 server), 185.199.108.1 (exfil destination). 2) Ports: 4444/TCP (reverse shell). 3) Behavioral signatures: SSH sweep pattern (sequential IPs, 50ms intervals), large outbound transfer ratio (450KB out / 2KB in on HTTPS), SYN-only probes with 5s timeout. 4) Network signatures: Outbound traffic to port 4444 from internal subnet, DNS queries to external resolver followed by large data transfer within 2 seconds. 5) Timing patterns: All activity within 4-minute window (02:00-02:04), suggesting automated tooling. YARA/Snort rules should flag: outbound 4444, internal SSH sweep >5 hosts in 10s, upload/download ratio >100:1 on single connection."},
    {"id": "anom_027", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Using the microservice metrics, estimate the blast radius of the incident. What percentage of total system capacity was affected?",
     "context": METRICS_DATA,
     "expected_answer": "Blast radius analysis: api-gateway: 100% affected (60% error rate, effectively down). Handles 500 req/s normally = primary traffic path. user-service: ~70% affected (12% error rate, 3x latency). Handles 200 req/s normally. payment-service: ~10% affected (2% error rate, minor latency increase). Handles 100 req/s normally. Total capacity: 800 req/s across all services. Affected capacity: 500 (api-gw fully) + 140 (70% of user-svc) + 10 (10% of payment) = 650 req/s impacted. Blast radius: 650/800 = 81% of total system capacity affected. However, since api-gateway is the entry point, effective user impact is likely 100% as all traffic routes through it."},
    {"id": "anom_028", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a fraud detection analyst with expertise in behavioral analytics. Analyze patterns across multiple customers.",
     "user_prompt": "For customer C100, determine the exact moment of account compromise based on transaction patterns. What was the attacker's likely strategy?",
     "context": FINANCIAL_DATA,
     "expected_answer": "Account compromise timeline: Legitimate activity: TXN001-003 (14:23-15:10) - normal retail purchases on mobile/desktop in US, amounts $12-$89. Compromise point: Between 15:10 and 15:12 (2-minute gap). Attack begins: TXN004 at 15:12. Attacker strategy: 1) Test with wire transfer to NG ($2500) - verify account access works for large amounts. 2) Immediately move to crypto ($1800 to RU) - harder to trace/reverse. 3) Final large wire ($3200 to NG) from 'unknown' device - maximum extraction before detection. Total stolen: $7,500 in 2 minutes. Strategy indicates experienced fraudster: high-value targets, multiple channels, speed over stealth."},
    {"id": "anom_029", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Predict when the api-gateway would have completely crashed if no intervention occurred, based on the degradation trajectory.",
     "context": METRICS_DATA,
     "expected_answer": "Extrapolating the degradation curve: Error rate progression: 0.4% -> 2.5% -> 8.5% -> 15.2% -> 25% -> 45% -> 60% (roughly doubling every 5 min initially, then linear). At this rate, 100% error rate would be reached around 10:55-11:00. CPU/Memory: Already at 99% by 10:40, so OOM kill or process crash likely within 5-10 minutes (by 10:50). Active connections: Growing ~50/5min, max file descriptors typically 1024 or 65535. At 980 and growing, ulimit exhaustion likely by 10:50-10:55. Prediction: Complete crash (OOM or connection exhaustion) by approximately 10:50-10:55 without intervention. The throughput drop to 150 req/s at 10:45 suggests the system was already in death spiral."},
    {"id": "anom_030", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a network security analyst specializing in threat hunting. Analyze traffic patterns to build an attack timeline.",
     "user_prompt": "Assess whether the HTTPS connection to 185.199.108.1 could be legitimate. What evidence supports or refutes this?",
     "context": NETWORK_DATA,
     "expected_answer": "Evidence for legitimate: 185.199.108.0/22 is GitHub's IP range, so this could be a git push or artifact upload. Port 443 (HTTPS) is standard. Evidence for suspicious: 1) The initial request (5KB) received 120KB response - normal for web browsing. 2) But the NEXT connection sent 450KB with only 2KB response - this is an upload pattern, not browsing. 3) Timing: occurs at 02:01 AM (unusual for developer activity). 4) Context: immediately followed by SSH scanning and C2 communication. Assessment: While the IP is GitHub's, the upload pattern at 2 AM combined with subsequent malicious activity strongly suggests exfiltration disguised as a git push. The attacker likely pushed stolen data to a repository they control."},
    {"id": "anom_031", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Calculate the Mean Time Between Failures (MTBF) and Mean Time To Recovery (MTTR) from the timeseries data. What SLA impact does this incident have?",
     "context": TIMESERIES_DATA,
     "expected_answer": "From this single day of data: Failure onset: 15:00 (first anomalous readings). Recovery: 17:00 (metrics return to normal). MTTR = 2 hours. Duration of impact: 2 hours out of 24 = 8.3% downtime for this day. SLA impact: If SLA promises 99.9% uptime (43.8 min/month allowed downtime), this 2-hour incident consumes 2.7x the entire monthly error budget in one event. If SLA is 99.95% (21.9 min/month), it's 5.5x the monthly budget. For 99.99% SLA (4.3 min/month), it's 27.9x over budget. This single incident would breach any SLA stricter than 91.7% monthly uptime, assuming no other incidents that month."},
    {"id": "anom_032", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a fraud detection analyst with expertise in behavioral analytics. Analyze patterns across multiple customers.",
     "user_prompt": "Design a real-time alerting threshold for the transaction monitoring system. Balance detection speed against false positive rate.",
     "context": FINANCIAL_DATA,
     "expected_answer": "Tiered alerting design: TIER 1 - Immediate Block (< 1 sec): Single transaction >$5000 to high-risk country, or >3 transactions totaling >$3000 in 5 minutes to different countries. Expected catch: C100 (blocked at TXN004). FPR: <1%. TIER 2 - Real-time Review (< 30 sec): Impossible travel (same card, 2 countries, <2 hours), or amount >5x rolling 7-day average. Expected catch: C201, C412. FPR: 3-5%. TIER 3 - Batch Analysis (< 5 min): Duplicate transaction patterns (same merchant/amount within 10 min), unusual time-of-day activity. Expected catch: C305. FPR: 10-15%. Tuning: Start conservative (more blocks), then relax based on customer complaints. Use ML model confidence to adjust thresholds per customer risk profile."},
    {"id": "anom_033", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Analyze the request_count pattern in the timeseries data. Is the traffic spike at 15:00-16:00 organic growth or an attack? Justify with statistical evidence.",
     "context": TIMESERIES_DATA,
     "expected_answer": "Statistical analysis of request patterns: Normal daily pattern shows gradual ramp (620 at 03:00 -> 4100 at 14:00), with hourly growth rate of ~315 req/hr during business hours. At 15:00: 4500 requests (expected ~4400 based on trend = only +2.3% above trend). At 16:00: 4800 requests (expected ~4700 = +2.1% above trend). The request count itself is NOT anomalous - it follows the natural daily curve. The anomaly is in the SYSTEM RESPONSE to normal-ish traffic: errors jumped 5x and latency 2x despite only marginal traffic increase. This suggests the root cause is NOT a traffic spike/DDoS but rather an internal degradation (memory leak, connection pool exhaustion, or bad deployment) that reduced system capacity, making normal traffic appear overwhelming."},
    {"id": "anom_034", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a senior SRE analyzing correlated anomalies across multiple metrics. Provide statistical reasoning.",
     "user_prompt": "Using the microservice metrics, determine if the user-service degradation is caused by the api-gateway or vice versa. What's the causal direction?",
     "context": METRICS_DATA,
     "expected_answer": "Causal analysis: Both services degrade at 10:20 simultaneously, making temporal ordering inconclusive. However, examining the degradation patterns: api-gateway requests INCREASE (500->1500) while user-service requests also increase (200->600). If api-gateway was failing, it would send FEWER requests downstream, not more. The pattern suggests: external traffic surge -> api-gateway forwards all requests -> user-service overwhelmed -> user-service slows down -> api-gateway connections pile up waiting for user-service responses -> api-gateway connection pool exhausts -> cascading failure. Evidence: api-gateway's active connections (200->980) grow faster than its request rate drops, indicating it's waiting on downstream responses. Causal direction: user-service saturation causes api-gateway failure through back-pressure."},
    {"id": "anom_035", "category": "anomaly_detection", "difficulty": "medium",
     "system_prompt": "You are a network security analyst specializing in threat hunting. Analyze traffic patterns to build an attack timeline.",
     "user_prompt": "Estimate the attacker's skill level and tooling based on the network traffic patterns. Is this an APT or opportunistic attack?",
     "context": NETWORK_DATA,
     "expected_answer": "Skill assessment: MODERATE (likely automated toolkit, not APT). Evidence for automated/scripted: 1) SSH scan at exact 1-second intervals (02:02:00-02:02:07) - human wouldn't be this precise. 2) Sequential IP targeting (100-107) rather than randomized - basic scanning tool. 3) Standard port 4444 for C2 - default Metasploit, not custom infrastructure. Evidence against APT: 1) No attempt to blend with normal traffic patterns. 2) All activity compressed into 4 minutes (APTs are patient). 3) Using well-known IOCs (port 4444, sequential scanning). 4) No encryption rotation or protocol tunneling. Assessment: Likely an automated post-exploitation framework (Metasploit/Cobalt Strike) run by a moderately skilled attacker or red team. An APT would use legitimate ports, randomized timing, and encrypted C2 channels."},


]


if __name__ == "__main__":
    save("anomaly_detection.json", prompts)
    print(f"Generated {len(prompts)} anomaly_detection prompts")
