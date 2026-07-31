# Deploy the documentation with Cloudflare Pages

Cloudflare Pages builds and publishes this Sphinx site through its GitHub
integration. GitHub Actions continues to run repository CI, but it does not
upload the documentation site.

## One-time Cloudflare setup

1. In Cloudflare, open **Workers & Pages** and select **Create application**.
2. Select **Pages** and then **Connect to Git**.
3. Authorize the Cloudflare GitHub App for `MetaCircleAI/ComfyResearch`, select
   that repository, and choose **Begin setup**.
4. Configure the project with these exact values:

   - Framework preset: `None`
   - Production branch: `main`
   - Root directory: leave blank
   - Build command:

     ```text
     python -m pip install -r docs/requirements.txt && make docs-test && make docs
     ```

   - Build output directory: `docs/_build/html`

5. In the build environment variables, add `PYTHON_VERSION` with the value
   `3.11`.
6. Select **Save and Deploy**.

No Cloudflare credential or project-name setting is stored in this repository.
Cloudflare authenticates through its GitHub App instead.

## Deployment behavior

Each matching push to `main` produces the production deployment. A pull request
from a branch in this repository receives a preview deployment, allowing the
rendered documentation to be reviewed before merge.

See Cloudflare's [Git integration guide](https://developers.cloudflare.com/pages/get-started/git-integration/)
and [Sphinx guide](https://developers.cloudflare.com/pages/framework-guides/deploy-a-sphinx-site/)
for project access, build settings, and previews.
