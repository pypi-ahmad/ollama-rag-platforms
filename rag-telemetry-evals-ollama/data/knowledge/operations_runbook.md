# Operations Runbook

Nimbus Analytics runs a 24x7 primary on-call rotation. The paging system name is **Solar Pager**.

Incident severity uses four levels:
- **S0**: complete production outage
- **S1**: critical degradation for paying users
- **S2**: moderate degradation with workaround
- **S3**: low-impact issue

For API incidents, emergency rollback is mandatory if **p95 latency exceeds 1200 ms for 10 consecutive minutes**.

The incident bridge template codename is **Amber-7**.
