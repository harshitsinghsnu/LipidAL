# Read-only Vercel deployment

**Published:** [https://lipidal.vercel.app](https://lipidal.vercel.app), in the
existing `lipidal` project under the user's Hobby scope. Public verification
without authentication checked all 667 exported asset paths, the PDF bytes,
highlighted structures, new-study filters, disabled cloud uploads, and mobile
layout. See `deployment_verification.json` for the exact verification snapshot.

The public build is a pre-generated, read-only export in `vercel_dashboard/`.
The root `vercel.json` supports GitHub import with **Root Directory `./`**,
**Framework Preset Other**, **Output Directory `vercel_dashboard`**, and empty
build/install commands. No environment variables are required. Deployment
status and the verified URL are recorded separately after publication.

## What can be published

The export contains 5,850 original benchmark views, 90 explanation reports,
9,720 new main campaigns, 1,944 adaptation comparisons (648 reused), the
updated manuscript, figure supplements, linked plots and derived result tables.
It excludes `results/user_runs`, raw source
datasets, pretrained model weights, full GP checkpoints and Python packages.
Only explicitly allowlisted report assets are copied. Review the derived
data/figure redistribution permissions and conference anonymity requirements
before public release. Do not deploy the entire repository or its data folder.

The read-only site does **not** accept uploaded CSVs or execute AL. It explains
how to start the local Python app; the Run button is disabled. Public users
cannot access your localhost server remotely. No credentials or private
uploads are required for static browsing.

## Rebuild and validate

From `D:\design\proteinlipid`:

```powershell
python export_vercel_dashboard.py
python test_vercel_export.py
```

The export is approximately 100 MB. `export_manifest.json` inventories and
hashes the report assets. The validation report is saved outside the export
as `vercel_export_validation.json`.

## Publish after approval and account connection

For Git deployment of this repository, retain project root **`./`**, Framework
Preset **Other**, and output directory **`vercel_dashboard`**. The root
configuration disables dependency installation/building; it publishes only the
reviewed static export. Alternatively, with an
authenticated Vercel CLI, run `vercel` from the repository root to obtain a preview;
review it before a production deployment. The local configuration supplies
basic response headers and does not define Python Functions.

The root-only patterns in `.vercelignore` intentionally start with `/` so
`/results/` excludes local results but not `vercel_dashboard/results/`.
Unanchored exclusions would omit public figures. `.vercel/` and `.env*`
remain private and ignored; never paste tokens into GitHub or the manuscript.

After publishing, verify the actual HTTPS URL, all dropdowns, plot links and
summary downloads. Only then add that URL to the manuscript and replace its
explicit statement that public deployment has not been verified. For a
double-blind submission, use an appropriately anonymous URL and metadata.

## Online AL requires a separate design

The existing Flask runner uses an in-memory thread pool and local files.
Moving it unchanged into request-scoped functions would not provide durable
campaign execution or job recovery. A hosted version needs authentication,
an explicit upload/data-retention policy, persistent object storage, a durable
queue and worker state, compute/resource limits, and an appropriately
provisioned Python worker (GPU optional for encoder acceleration).

This is an architectural limitation of the current runner, not a claim that
Vercel cannot host Python. Current [Vercel Functions documentation](https://vercel.com/docs/functions/limitations)
describes duration, memory and bundle constraints, including beta extensions;
plan/runtime eligibility must be checked for an actual backend design.
See also [Vercel Git deployment instructions](https://vercel.com/docs/git).
