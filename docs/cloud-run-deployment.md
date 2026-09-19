# Deploy the NebulaX Control Room to Google Cloud Run

This guide deploys the repository's Python web app as a private Cloud Run service. It uses
Google Cloud Build, so Docker does not need to be installed on the operator's computer. The
container image is stored in Artifact Registry.

The commands below assume Bash or a compatible shell and must be run from the repository root.
Use a permanent Google Cloud project for a lasting deployment; training or Qwiklabs projects
can expire and remove their resources.

## Before deploying

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI (`gcloud`);
- permission to enable APIs, run Cloud Build, create an Artifact Registry repository, deploy
  Cloud Run services, and act as the service identity; and
- the four selected model files and their `active_model.json` manifests in this checkout.

See Google's role lists in the
[Artifact Registry guide](https://docs.cloud.google.com/artifact-registry/docs/docker/store-docker-container-images)
and [Cloud Run deployment guide](https://docs.cloud.google.com/run/docs/deploying) when using a
project that you do not own.

Never commit access tokens, downloaded service-account keys, or the local `gcloud` credential
directory. For an interactive deployment, authenticate in the terminal:

```sh
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth list
gcloud config get-value project
```

Confirm that the active account and project are the intended billing targets before continuing.

## 1. Set deployment names

Choose a unique image tag for every build. Do not reuse a tag when producing a new revision.

```sh
export NEBULAX_PROJECT_ID="YOUR_PROJECT_ID"
export NEBULAX_REGION="us-central1"
export NEBULAX_REPOSITORY="nebulax-repo"
export NEBULAX_SERVICE="nebulax-control-room"
export NEBULAX_TAG="v1"
export NEBULAX_IMAGE="${NEBULAX_REGION}-docker.pkg.dev/${NEBULAX_PROJECT_ID}/${NEBULAX_REPOSITORY}/control-room:${NEBULAX_TAG}"

gcloud config set project "${NEBULAX_PROJECT_ID}"
```

Use the same region for Artifact Registry and Cloud Run unless there is a specific reason not to.

## 2. Check the application

Run the app contract tests before building:

```sh
python3 -m unittest discover -s app/tests -v
```

If Node.js is installed, also check the browser script:

```sh
node --check app/static/app.js
```

Review the Cloud Build upload set:

```sh
gcloud meta list-files-for-upload
```

The list should contain only the Docker configuration, runtime source, static UI, preview CSVs,
and the four selected model artifacts/manifests. `.gcloudignore` is an explicit allowlist. When a
new runtime module is introduced, add it there as well as to the `Dockerfile`; otherwise Cloud
Build will not receive it. Raw datasets, notebooks, tests, reports, and experimental weights must
not appear in the upload list.

## 3. Enable the services

```sh
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  --project "${NEBULAX_PROJECT_ID}"
```

Check whether the Docker repository already exists:

```sh
gcloud artifacts repositories describe "${NEBULAX_REPOSITORY}" \
  --location "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}"
```

If that command reports `NOT_FOUND`, create the repository once:

```sh
gcloud artifacts repositories create "${NEBULAX_REPOSITORY}" \
  --repository-format docker \
  --location "${NEBULAX_REGION}" \
  --description "NebulaX Control Room container images" \
  --project "${NEBULAX_PROJECT_ID}"
```

## 4. Build the image

```sh
gcloud builds submit \
  --tag "${NEBULAX_IMAGE}" \
  --project "${NEBULAX_PROJECT_ID}" \
  .
```

Wait for `STATUS: SUCCESS`. Record the build ID, image tag, and reported `sha256` digest with the
deployment evidence. Cloud Run resolves the selected image when creating the revision; using a
new tag for each build keeps the history understandable.

## 5. Deploy the private service

```sh
gcloud run deploy "${NEBULAX_SERVICE}" \
  --image "${NEBULAX_IMAGE}" \
  --region "${NEBULAX_REGION}" \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10 \
  --min 0 \
  --max 1 \
  --no-allow-unauthenticated \
  --project "${NEBULAX_PROJECT_ID}"
```

`--min 0` permits scale-to-zero. `--max 1` is deliberate: the current app stores runs and download data in memory and an
instance-local SQLite database. Restricting normal operation to one instance avoids most
cross-instance history inconsistencies and caps cost, but it does not make history durable.
Cloud Run can briefly exceed a configured maximum, and an instance restart or scale-to-zero
event clears all run history. Download results and technical records promptly. A genuinely
multi-instance deployment requires external session/result storage.

The service remains private because of `--no-allow-unauthenticated`. Do not change this merely to
make the URL easier to open.

## 6. Verify the deployment

Get the canonical URL and active revision:

```sh
export NEBULAX_URL="$(gcloud run services describe "${NEBULAX_SERVICE}" \
  --region "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}" \
  --format 'value(status.url)')"

gcloud run services describe "${NEBULAX_SERVICE}" \
  --region "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}" \
  --format 'yaml(status.url,status.latestReadyRevisionName,status.traffic)'
```

An unauthenticated request should be rejected with HTTP `403`:

```sh
curl -i "${NEBULAX_URL}/"
```

An authenticated model-status request should return HTTP `200` and show `"ready": true` for
Door, ACV, Rail Corrugation, and SHM:

```sh
curl -i \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  "${NEBULAX_URL}/api/status"
```

This proves that the revision starts, serves HTTP, enforces IAM, and can read the deployed model
manifests and checksums. It does not prove live inference. Complete a known-input smoke test in
the browser and compare its downloaded CSV with a trusted local result before relying on a new
deployment.

## 7. Open the private app in a browser

A private `run.app` URL does not automatically redirect a normal browser to Google sign-in. Use
Google's authenticated local proxy:

```sh
gcloud run services proxy "${NEBULAX_SERVICE}" \
  --region "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}" \
  --port 8080
```

If prompted, allow `gcloud` to install the `cloud-run-proxy` component. Keep the command running
and open <http://127.0.0.1:8080>. Stop the proxy with `Ctrl-C`.

The proxy address is local, but requests and uploaded files are forwarded to Cloud Run. To keep
sensor files entirely on the operator's computer, run `python app/server.py` instead.

For direct browser access with an organization sign-in page, configure a supported HTTPS load
balancer and Identity-Aware Proxy separately. That infrastructure is outside this repository's
basic deployment.

## Cloud upload limit

The local app accepts files up to 120 MB and sends queued files individually. This container uses a Python HTTP/1 server, while
Cloud Run limits HTTP/1 requests to 32 MiB. Multipart encoding adds overhead, so keep each cloud
upload comfortably below that limit (for example, below 30 MiB) and use smaller individual recordings. See the current [Cloud Run quotas and limits](https://docs.cloud.google.com/run/quotas)
before deployment.

## Deploy an update

1. Run the tests.
2. Choose a new immutable tag, such as `v2`, and update `NEBULAX_TAG` and `NEBULAX_IMAGE`.
3. Review `gcloud meta list-files-for-upload` again.
4. Run the build command.
5. Run the deploy command with the new image.
6. Confirm that the new revision receives 100% of traffic and repeat the authenticated checks.

List revisions with:

```sh
gcloud run revisions list \
  --service "${NEBULAX_SERVICE}" \
  --region "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}"
```

To roll back, route traffic to a known-good revision:

```sh
gcloud run services update-traffic "${NEBULAX_SERVICE}" \
  --to-revisions KNOWN_GOOD_REVISION=100 \
  --region "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}"
```

## Troubleshooting

- **OAuth page reports an error:** rerun `gcloud auth login --no-launch-browser --force` and use
  the newly generated URL. Never paste a password or private key into chat or documentation.
- **Build uploads unexpected files:** stop and fix `.gcloudignore` before submitting. The
  `.dockerignore` file controls the later Docker build context, not the source archive initially
  sent to Cloud Build.
- **Container does not become ready:** inspect the revision logs. The server must listen on
  `0.0.0.0` and the `PORT` supplied by Cloud Run; `app/server.py` and the `Dockerfile` already
  provide this behavior.
- **Direct service URL returns `403`:** this is expected for the private service. Use the
  authenticated proxy or an identity token.
- **A proxy POST returns `403`:** deploy the current `app/server.py`. It explicitly permits HTTP
  loopback origins while continuing to reject unrelated origins.
- **Model status is not ready:** compare each artifact with its `active_model.json` checksum and
  rebuild only from trusted model files.
- **Large uploads fail before the app responds:** split the batch below Cloud Run's HTTP/1
  request limit.

## Cleanup

Deleting resources is irreversible and removes the deployed service or stored images. Confirm
the project and names before running either command:

```sh
gcloud run services delete "${NEBULAX_SERVICE}" \
  --region "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}"

gcloud artifacts repositories delete "${NEBULAX_REPOSITORY}" \
  --location "${NEBULAX_REGION}" \
  --project "${NEBULAX_PROJECT_ID}"
```
