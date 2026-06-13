# Release Control Policy

Release train window is **Tuesday 16:00 UTC**.

Each deployment starts with a canary on **8% of traffic**.

Promotion to 100% is allowed after **45 minutes** if error rate remains below **0.6%**.

Promotion is paused if any S0/S1 incident is active.
