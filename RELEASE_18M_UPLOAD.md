# LumoLM 18M release upload

This release package is 217,829,686 bytes, split into 11 files so they can be uploaded through the GitHub web UI without a single huge upload.

## Upload the parts

1. Download all 11 `.partNN` files from the ChatGPT conversation.
2. In this repository, open **Add file → Upload files**.
3. Create/use the directory `release-parts` and upload all 11 parts together. Keep every filename exactly as supplied.
4. Commit the upload to `main`.

Expected filenames:

```text
LumoLM-18M-20261024-Complete.zip.part01
LumoLM-18M-20261024-Complete.zip.part02
LumoLM-18M-20261024-Complete.zip.part03
LumoLM-18M-20261024-Complete.zip.part04
LumoLM-18M-20261024-Complete.zip.part05
LumoLM-18M-20261024-Complete.zip.part06
LumoLM-18M-20261024-Complete.zip.part07
LumoLM-18M-20261024-Complete.zip.part08
LumoLM-18M-20261024-Complete.zip.part09
LumoLM-18M-20261024-Complete.zip.part10
LumoLM-18M-20261024-Complete.zip.part11
```

## Let GitHub Actions synthesize and publish

1. Open **Actions → Assemble and publish LumoLM 18M release**.
2. Click **Run workflow**, select `main`, and keep the default tag/title unless a different version is intended.
3. The workflow concatenates all parts, verifies exact size and SHA-256, runs `unzip -t`, and only then creates the GitHub Release and attaches the ZIP.
4. If a part is missing or corrupted, the workflow fails before publishing.

Expected archive size: `217829686` bytes.

Expected SHA-256:

```text
e4f0ecc1c0042323dfb072626d6a4d59709996b383604f7eee0a19c9f84e9ae9
```

## Experimental status

The package passed ZIP integrity and a basic CPU smoke test. The current 54-question raw-neural benchmark was 8/54, so this preview must not be described as a demonstrated improvement in general answer quality.
