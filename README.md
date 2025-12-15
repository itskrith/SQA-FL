# SQA-FL: Secure, Quality-Aware Federated Learning for Clinical Analytics

This project implements a prototype **Secure, Quality-Aware Federated Learning (SQA-FL)** framework
for privacy-preserving clinical analytics, as described in the course project proposal.

## Features

- **Virtual hospitals** simulated on a public tabular health dataset (UCI heart failure).
- **Centralized training** baseline (all data pooled).
- **Vanilla Federated Learning (FedAvg)** with non-IID partitions.
- **Quality-aware FL (SQA-FL)** using gradient-alignment–based quality scores per client.
- **Homomorphic encryption (Paillier)** for client quality scores:
  - Server only observes encrypted scores and the decrypted *global sum*.
- **Secure-aggregation-style update aggregation**:
  - Server only keeps the **aggregate scaled update**, not per-client updates.
- **Client dropout robustness**:
  - Training still converges when ~30% of clients randomly drop out each round.

## Project Structure

```text
sqa_fl/
  sqa_fl/
    __init__.py
    data.py          # dataset loading & federated partitioning
    models.py        # MLP model definition
    crypto/
      __init__.py
      paillier_he.py # Paillier keygen, encrypt/decrypt for quality scores

  experiments/
    __init__.py
    run_centralized.py       # centralized baseline
    run_fedavg_manual.py     # vanilla FedAvg (manual loop, no ray needed)
    run_sqa_fl_quality.py    # quality-weighted FL (no HE)
    run_sqa_fl_he.py         # quality-weighted FL + HE over quality scores
    run_sqa_fl_he_dropout.py # same as above, with ~30% client dropout

  requirements.txt
  README.md