# Release Protocol

Production releases are aligned to the **Tuesday 16:00 UTC** release train.

Every release starts with a canary deployment to **8%** of traffic.

Promotion to full rollout is allowed after **45 minutes** when request error rate remains below **0.6%**.

If any S0/S1 incident is active, release promotion is paused automatically.
