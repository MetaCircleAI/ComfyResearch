# Application screenshot capture plan

Use this checklist after the documentation structure and copy are stable.
Capture each image manually from the running application. Every capture should
use a 1440 x 900 CSS-pixel viewport, the light theme, and a clean demo project.
Crop to the smallest region that still preserves the relationship the image is
meant to explain. Export PNG files at the paths below.

Do not capture real hostnames, usernames, passwords, SSH keys, tokens, private
paths, unpublished results, or personal project names. Use
`gpu.example.invalid` wherever a remote host must be visible.

## IMG-01: Stable workbench overview

- **Page:** Introduction
- **Purpose:** Orient a new reader to the stable workbench before terminology is introduced.
- **Capture:** A clean project with the node library, a small connected graph, the Trainer, one Observable, and its result visible.
- **Crop:** Keep the full application frame, including the project rail and canvas; omit browser chrome.
- **Final file:** `docs/en/_images/app/overview-stable-workbench.png`
- **Alt text:** Comfy Research workbench showing a project, a connected training graph, Trainer controls, and an Observable result.
- **Sensitive-data review:** Use only bundled examples and synthetic result names.

## IMG-02: Select the first-graph template

- **Page:** Get Started - Run your first graph
- **Purpose:** Show where a beginner selects a bundled graph template.
- **Capture:** The Templates view with the recommended first-run template selected.
- **Crop:** Include the Templates navigation label, selected template row, and load action.
- **Final file:** `docs/en/_images/app/first-graph-template.png`
- **Alt text:** Templates view with the recommended first-run graph template selected.
- **Sensitive-data review:** Use only a bundled template and an empty demo project.

## IMG-03: Configure Trainer for a CPU run

- **Page:** Get Started - Run your first graph
- **Purpose:** Make the minimum valid Trainer setup and its graph connections unambiguous.
- **Capture:** The loaded graph with the Trainer selected and CPU configured.
- **Crop:** Include the Trainer node, its connected inputs, and the device control; keep labels readable.
- **Final file:** `docs/en/_images/app/first-graph-trainer.png`
- **Alt text:** A connected Trainer node configured to run the example graph on CPU.
- **Sensitive-data review:** Confirm that no local file paths or private dataset names are visible.

## IMG-04: Confirm a completed first run

- **Page:** Get Started - Run your first graph
- **Purpose:** Give the reader a visual success criterion after running the graph.
- **Capture:** The completed run state with the result area and one Observable visualization visible.
- **Crop:** Include the completion state, result name, and chart; omit unrelated panels.
- **Final file:** `docs/en/_images/app/first-graph-results.png`
- **Alt text:** Completed example run with a recorded Observable displayed as a chart.
- **Sensitive-data review:** Use synthetic output and a fresh local project.

## IMG-05: Add a node from the library

- **Page:** User Guide - Build and run graphs
- **Purpose:** Show the spatial relationship between the node library and canvas.
- **Capture:** Node search filtered to a common stable node while the target canvas is visible.
- **Crop:** Include the search field, matching row, and enough canvas to show the drop target.
- **Final file:** `docs/en/_images/app/add-node-from-library.png`
- **Alt text:** Node library search beside a canvas where the selected node can be added.
- **Sensitive-data review:** Use a built-in node and a generic project name.

## IMG-06: Complete stable training graph

- **Page:** User Guide - Build and run graphs
- **Purpose:** Explain compatible sockets and the minimum stable training core as one system.
- **Capture:** A complete graph containing data, model, optimizer, loss, and Trainer nodes with readable connections.
- **Crop:** Fill the frame with the graph; preserve node titles and socket colors without excessive empty canvas.
- **Final file:** `docs/en/_images/app/stable-training-core.png`
- **Alt text:** Complete training graph connecting data, model, optimizer, loss, and Trainer nodes.
- **Sensitive-data review:** Use bundled nodes, synthetic data, and no local paths.

## IMG-07: Connect an Observable to its visualization

- **Page:** User Guide - Observables
- **Purpose:** Distinguish the value being recorded from the visualization used to inspect it.
- **Capture:** An Observable connected in the graph with its corresponding result visualization open.
- **Crop:** Include the Observable edge and paired visualization in one frame.
- **Final file:** `docs/en/_images/app/observable-and-visualization.png`
- **Alt text:** Observable connected to a training graph with its recorded values shown in a visualization.
- **Sensitive-data review:** Use a bundled scalar Observable and synthetic values.

## IMG-08: Project tabs and their canvases

- **Page:** User Guide - Projects and artifacts
- **Purpose:** Show that each project owns one canvas and that users switch projects through project tabs.
- **Capture:** Baseline and Comparison project tabs while one project's canvas is open.
- **Crop:** Include both project tabs and the open canvas title.
- **Final file:** `docs/en/_images/app/project-canvas-tree.png`
- **Alt text:** Baseline and Comparison project tabs with the Baseline project's single canvas open in the workbench.
- **Sensitive-data review:** Use the project names Baseline and Comparison.

## IMG-09: Choose an artifact save tier

- **Page:** User Guide - Projects and artifacts
- **Purpose:** Show the decision point between the available save scopes without turning prose into UI directions.
- **Capture:** The save dialog or menu with the supported artifact tiers visible.
- **Crop:** Keep the tier names, short descriptions, and confirmation action.
- **Final file:** `docs/en/_images/app/save-artifact-tiers.png`
- **Alt text:** Save dialog showing the available artifact persistence tiers.
- **Sensitive-data review:** Confirm destination paths and personal project names are not visible.

## IMG-10: Configure a sanitized remote GPU target

- **Page:** User Guide - Local and remote compute
- **Purpose:** Clarify which remote fields belong together while reinforcing the security warning in the text.
- **Capture:** Remote training configuration populated with fake values and password authentication left empty.
- **Crop:** Include host, user, remote path, Python, identity file, and upload setting.
- **Final file:** `docs/en/_images/app/remote-gpu-configuration.png`
- **Alt text:** Remote GPU configuration using a fake host, SSH identity file, remote path, and Python command.
- **Sensitive-data review:** Host must be `gpu.example.invalid`; use fake user `researcher`; show no password, real key path, or token.

## IMG-12: Compare CBS and CLR reproduction curves

- **Page:** Examples - Jastrzebski Figure 1 reproduction
- **Purpose:** Let readers compare the expected scientific result with their own run.
- **Capture:** The canonical CBS and CLR curves produced by the documented reproduction workflow.
- **Crop:** Keep axes, legend, units, and the complete curves; remove notebook or desktop chrome.
- **Final file:** `docs/en/_images/app/jastrzebski-cbs-clr-curves.png`
- **Alt text:** Reproduction plot comparing constant batch size and cyclical learning-rate training curves.
- **Sensitive-data review:** Confirm labels and metadata contain no local paths or private run identifiers.

## IMG-13: Compare small-batch and large-batch accuracy

- **Page:** Examples - Keskar Figures 2 and 3 reproduction
- **Purpose:** Provide the first expected-result checkpoint for the reproduction.
- **Capture:** The final small-batch and large-batch accuracy comparison generated by the documented run.
- **Crop:** Keep axes, legend, units, and confidence or run annotations that are required to interpret the result.
- **Final file:** `docs/en/_images/app/keskar-sb-lb-results.png` (combined Figures 2 and 3 capture)
- **Alt text:** Reproduction plot comparing final accuracy for small-batch and large-batch training.
- **Sensitive-data review:** Use only reproducible public experiment labels and sanitized run metadata.

## IMG-14: Inspect Keskar interpolation sharpness

- **Page:** Examples - Keskar Figures 2 and 3 reproduction
- **Purpose:** Provide the second expected-result checkpoint and make the sharpness comparison visually testable.
- **Capture:** The interpolation or loss-surface profile for the small-batch and large-batch solutions.
- **Crop:** Keep axes, legend, sampling range, and both complete profiles.
- **Final file:** `docs/en/_images/app/keskar-sb-lb-results.png` (combined Figures 2 and 3 capture)
- **Alt text:** Interpolation profile comparing sharpness around small-batch and large-batch solutions.
- **Sensitive-data review:** Remove local artifact paths, machine names, and private run identifiers.

## IMG-15: Verify the edge-of-stability CPU trajectory

- **Page:** Examples - Edge of Stability on a Small CPU MLP
- **Purpose:** Show the leading Hessian eigenvalue tracking the $2 / \eta$
  cutoff beside the non-monotonic, net-decreasing training loss.
- **Capture:** A completed 80-update CPU run with `Training viz` and `Hessian
  viz` visible together.
- **Crop:** Keep both full trajectories, axes, legend, $\lambda_1$,
  $\lambda_2$, and the dashed $2 / \eta = 10$ cutoff; remove browser chrome.
- **Final file:** `docs/en/_images/app/edge-of-stability-cpu.png`
- **Alt text:** Training loss and the top two Hessian eigenvalues from a CPU
  edge-of-stability run, with a dashed sharpness cutoff at 10.
- **Sensitive-data review:** Use the bundled template and synthetic results
  only; do not show local paths, project names, or private run identifiers.
