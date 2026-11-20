from cardiac_dynamics.experiment import evaluate,signal
def test_signal_reproducible(): assert (signal(30,2)==signal(30,2)).all()
def test_metrics_valid(): assert all(v["rollout_rmse"]>=0 for v in evaluate(3).values())

