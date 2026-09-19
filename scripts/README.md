# Supporting scripts

Run commands from the repository root using the inference environment.

```sh
python scripts/package.py --dest C151-new
```

The packaging script assembles a separate submission directory containing the app, selected models, `predictions.zip`, and technical write-ups under `Optional_Items/`. It reads the write-ups from `docs/writeups/` and refuses to overwrite an existing destination. Add the required demo video to the resulting submission directory after recording it.
