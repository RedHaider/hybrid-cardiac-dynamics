# Hybrid Cardiac Dynamics

Original synthetic benchmark exploring hybrid neural-physics cardiac modeling.
It forecasts an oscillating cardiac signal using a
physics prior, a learned autoregressive correction, and a hybrid of both.
No patient data are used; this is not a clinical tool.

Run: `python -m cardiac_dynamics.experiment --output outputs/demo`

Outputs compare rollout RMSE for physics-only, data-only, and hybrid models.
