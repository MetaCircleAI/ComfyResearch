---
doc_type: how-to
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Use a remote GPU

:::{div} cr-article-lead
Keep the application local, send a validated training graph to an existing SSH
host, and stream progress back to the same Trainer.
:::

## Before configuring remote execution

First complete a CPU run with the local service bound to `127.0.0.1`. Remote
execution adds SSH, environment, dependency, and data-transfer failure modes;
it should not be the first test of graph correctness.

The remote machine must already be running and reachable over SSH. Comfy
Research does not provision or stop the instance.

## Configure the Trainer

1. In the Trainer, choose **AutoDL GPU** as the compute device.
2. Enter the SSH host and user, remote repository path, and remote Python.
3. Prefer an SSH identity file. Leave the password field empty when key-based
   authentication is available.
4. Enable dataset upload only when the graph needs local data that is absent on
   the remote host.
5. Run the Trainer. The first request bootstraps the remote runtime, validates
   the graph, and then starts streamed training.

:::{figure} ../_images/app/remote-gpu-configuration.png
:alt: Remote GPU configuration using a fake host, SSH identity file, remote path, and Python command.
:class: cr-product-screenshot cr-product-screenshot-portrait
:::

## Protect credentials

Trainer remote settings are saved under
`.comfyresearch/remote_train_config.json`. This file is Git-ignored, but any
entered password is stored as plain JSON on the local machine. Git ignore does
not encrypt it or protect copies made by backups and support bundles.

Prefer, in order:

1. SSH key authentication with a restricted identity file;
2. environment variables supplied by a local secret manager;
3. a saved password only on a controlled workstation when the other methods
   are unavailable.

Never place real credentials in a graph, screenshot, issue, committed config,
or documentation example. Remove the saved config before sharing a repository
archive.

## Verify the run boundary

At the start of remote training, the backend synchronizes the required source
bundle when its digest changed, checks dependencies, validates the graph, and
streams NDJSON events over SSH. Pause and abort are cooperative remote control
requests; loss of SSH can interrupt control even if the remote process still
exists.

Treat the result like any other reproduction: record the remote Python and
package environment, accelerator type, graph artifact, seeds, and dataset
provenance. A successful remote stream proves transport and execution, not
equivalence with a local numeric result.

See [Application reference](../reference/application.md) for the remote
environment variables and precedence rules.
