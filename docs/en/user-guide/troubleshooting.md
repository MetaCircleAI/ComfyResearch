---
doc_type: how-to
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Troubleshoot a run

:::{div} cr-article-lead
Start from the visible symptom, reduce the graph to the smallest failing path,
and separate graph correctness from device and measurement cost.
:::

## The application does not start

1. Activate the expected Python environment.
2. Rebuild stale frontend output with `npm --prefix frontend run build`.
3. Start on an unused loopback port:

   ```bash
   python app.py --host 127.0.0.1 --port 8042
   ```

4. Request `http://127.0.0.1:8042/api/health` before opening the workbench.

If Vite renders but data does not load, confirm that the backend is running and
that `COMFYRESEARCH_PORT` matches its port.

## The Trainer rejects the graph

Work upstream from the Trainer:

1. Confirm every required socket has one intended source.
2. Check dataset and model dimensions.
3. Check that the loss accepts the model output and target type.
4. Check batch size against the training set.
5. Remove optional schedules and Observables until the core graph validates.

Do not reconnect edges by appearance alone. The source and target handles are
part of the executable graph contract.

## Training is unexpectedly slow or runs out of memory

- switch to CPU for a tiny correctness test or to the intended accelerator for
  the real workload;
- increase log frequency to record expensive measurements less often;
- disable Hessian, spectral, representation, or attention Observables;
- reduce model, dataset, and batch size independently;
- check whether comma-separated fields expanded into a Cartesian product.

## A visualization is empty

Confirm that the run reached a logged step, the Trainer output is connected to
the intended visualization, and the Observable has a paired result consumer.
An empty chart can be a routing or logging issue even when training completed.

## Remote execution fails

Verify ordinary SSH access outside the application, then check host, user,
remote path, remote Python, and authentication. Use the local CPU graph first.
Never paste the contents of `.comfyresearch/remote_train_config.json` into a
public report because it may contain a plain-text password.

For exact routes and event types, use the [Reference](../reference/index.md).
