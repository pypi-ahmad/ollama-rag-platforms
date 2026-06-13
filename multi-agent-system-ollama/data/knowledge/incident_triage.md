# Incident Triage Guide

Primary on-call paging is handled by **Solar Pager**.

Incident severities:
- **S0** full production outage
- **S1** critical customer-impacting degradation
- **S2** moderate degradation with workaround
- **S3** low-impact issue

For API incidents, emergency rollback is mandatory when **p95 latency exceeds 1200 ms for 10 consecutive minutes**.
