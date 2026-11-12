# ModelRunner for ComfyUI

Run hundreds of image, video, audio and 3D AI models from ComfyUI with one
[ModelRunner](https://modelrunner.ai/comfyui) API key and one balance. No local GPU and
no model downloads: each node sends its inputs to ModelRunner and returns the
result to your graph.

- **One typed node per model.** Every image, video, audio and 3D model in the
  catalog gets its own node with real widgets: dropdowns for options, number
  fields with their limits, image and audio sockets for file inputs. Find them
  under **ModelRunner** in the node menu, or search a model's name.
- **ModelRunner Run (any model)** runs any model by its `owner/alias` endpoint
  with a JSON input, including models newer than this release. Language models
  take an OpenAI-style body (`{"messages": [{"role": "user", "content": "..."}]}`)
  and return the reply in `output_json`.
- **ModelRunner Upload Image** turns an IMAGE into a URL for any URL input, and
  **ModelRunner Download Image** turns URLs back into an IMAGE.

Browse the catalog, with prices and examples, at https://modelrunner.ai/models.

## Install

From **ComfyUI Manager**, search for "ModelRunner" and install. Or by hand:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/modelrunner/modelrunner-comfyui
```

The pack needs nothing beyond what ComfyUI already installs. Use ComfyUI 0.12.1
or newer: older versions still load every node, but video and 3D nodes then
output only file paths and URLs.

## Your API key

Create a key at https://modelrunner.ai/settings/api-keys and add a balance. Then
either set it in the environment before starting ComfyUI:

```bash
export MODELRUNNER_API_KEY=your-key
```

or put the key alone in a file named `modelrunner_api_key.txt` in this pack's
folder under `ComfyUI/custom_nodes/`. That file is git-ignored, but an update
through ComfyUI Manager can replace the folder, so the environment variable is
the sturdier choice.

The key is never a node widget. Widget values are saved into workflow files and
into the metadata of every PNG ComfyUI writes, so a key there would leak each
time you shared an image.

## Using the nodes

1. Add a model's node, connect an image, audio or video socket if it has one,
   and set the widgets. Optional sockets can stay unconnected. A video socket
   takes a VIDEO from Load Video or another node; to pass a video URL instead,
   leave the socket empty and set it in `extra_json`, e.g.
   `{"video_url": "https://..."}`.
2. Queue the prompt. The console prints the request ID and the model's average
   cost per run; a progress bar shows while the run is queued and working.
3. Image nodes output an IMAGE batch, a MASK (from the alpha channel, so
   background-removal results work like LoadImage's), the output URLs and the
   request ID. Video, 3D and audio nodes download every result into
   `output/modelrunner/` and output the first result as a typed value (VIDEO
   for Save Video, a 3D file for Save 3D Model or Preview 3D, AUDIO for Save
   Audio), plus every file path and URL. Multi-value outputs hold one entry per
   line. Video, 3D and audio nodes run even with nothing connected after them.

Four example workflows ship with the pack and appear in ComfyUI's template
browser under this pack's name: text to image, an image mask from a loaded
image, image to video, and a language model through the run-any-model node.

Fields that have no widget (lists of objects, such as LoRA lists) can be set in
the node's `extra_json` box as a JSON object. It is merged last, so it also
overrides any widget.

Each node's tooltip shows the model's page and its average cost per run when
this pack was released. Current prices are on the model page.

## Costs and caching

**Every run is a paid request.** Read these before queueing big batches:

- ComfyUI skips a node whose inputs did not change and reuses its last result.
  Seed widgets start on **fixed**, so re-queueing the same graph costs nothing.
- Switching a seed to randomize, or turning **force_rerun** on, makes every queue a
  new paid run.
- **Interrupting stops the wait, not the job.** The run already submitted keeps
  going and is charged if it completes. Find it later by its request ID in your
  dashboard at https://modelrunner.ai.
- A failed run is not charged.
- The cost printed before a run is the model's recent average. Many video and
  audio models charge per second of output, so a longer clip (more frames, or a
  lower fps for the same frames) costs more than that average.

## Privacy

Images, audio and video connected to a node are uploaded so the model can fetch them.
Each upload is stored on ModelRunner's media host under an unguessable URL,
and so are the results. Do not connect anything you would not upload to a cloud
service.

## Models that are retired

A model that leaves the catalog keeps its node, moved under
**ModelRunner/Retired**, so saved workflows still open. Running it raises an
error that names the model to switch to.

## Updating the catalog (maintainers)

The typed nodes are generated when ComfyUI starts, from the committed
`catalog.json`. To refresh it:

```bash
python3 scripts/snapshot_catalog.py
python3 scripts/check_pack.py
```

The check fails on a duplicate node ID, a required field without a widget, or a
word from the private deny-list in any file git would publish, and lists every
field that only `extra_json` can set. The deny-list is not in the repo: set
`MODELRUNNER_DENYLIST` or create `scripts/denylist.txt` (git-ignored). CI reads
the `MODELRUNNER_DENYLIST` repository secret.

To release, bump `version` in `pyproject.toml` (and `VERSION` in
`modelrunner_nodes/client.py`), commit, then push a matching tag (`v0.2.1`).
The tag publishes to the Comfy Registry. `.comfyignore` keeps `scripts/`, CI
files and `assets/` out of the published zip, so only the files ComfyUI loads
are scanned and shipped.

## License

MIT
