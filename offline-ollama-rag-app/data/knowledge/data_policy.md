# Data Governance Policy

The analytics platform stores raw event logs for **400 days**.

Customer PII fields are encrypted with envelope keys under the vault policy alias **vault-river**.

Full logical backups run every **Sunday at 02:30 UTC** and are retained for 30 days.

All ad hoc analyst exports must be approved by a data steward and expire within 7 days.
