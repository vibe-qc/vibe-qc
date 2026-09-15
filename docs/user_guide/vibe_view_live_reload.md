# Live reload (auto-reload on file change)

vibe-view can watch the file it has open and hot-reload the scene whenever
the file changes on disk. This is the viewer-side half of live result
streaming: point it at the QVF a running calculation keeps rewriting (a
`checkpoint_qvf`, a growing optimization trajectory) and watch the results
arrive without re-opening anything.

## Turning it on

Flip the **Auto-reload on file change** switch in the **Display** card of
the left panel. The active file is then polled in the background (every
5 s by default) and reloaded whenever its *content* actually changes.

## What "changes" means

The watcher is deliberately conservative. It never hands the viewer a
half-written or unchanged file:

- **Content, not timestamps.** The file is fingerprinted from its QVF
  manifest (a digest per section). A `touch`, or a rewrite that produces
  identical content, does not reload.
- **Settle delay.** After the file starts moving, the watcher waits until
  it has stopped moving (0.5 s of quiet) before reading it, so a writer
  streaming a large archive isn't observed mid-copy.
- **Mid-write hold.** If the file was a valid QVF and momentarily isn't
  (a zip's central directory lands last), the watcher holds and retries
  rather than firing on a corrupt archive.

## What gets reloaded

The reload is scoped by a per-section diff of the manifest:

- **Only 2D-panel sections changed** (SCF history growing, a spectrum
  updating, new symmetry analysis): just the open panel refreshes. The 3D
  scene, your camera, and your section selection are untouched.
- **Anything structural** (atoms moved, sections added or removed, a
  volume or trajectory changed): the full scene rebuilds, but your
  current camera is kept (the file's camera bookmark is only applied when
  you *open* or *switch* files), and the section you had open is restored
  if it still exists.

The status bar reports each reload with the sections that moved. If you are
parked on the last frame of a growing trajectory, the reload keeps you on
the head as new frames stream in, so a live optimization plays forward on
its own.

## Streaming a running job

A vibe-qc job started with `checkpoint_qvf=…` rewrites a live QVF snapshot
as it runs (`provenance.run_status`, a monotonic `provenance.checkpoint`,
and `partial` on still-growing sections; see the
[QVF consumer reference](../consumer_qvf_reference.md)
section "Live / streaming checkpoints"). vibe-view surfaces that stream:

- **Watch live from the Job Manager.** A running job with a checkpoint
  shows a **Watch live** action (the eye icon). Clicking it opens the
  job's checkpoint QVF and turns auto-reload on for you, with no need to know
  the path. Submitting a job from the viewer streams a checkpoint by
  default (the **Stream live checkpoints** switch in the submit dialog), so
  a job you just launched is immediately watchable.
- **The app-bar chip** reflects the job status: it pulses **running**
  (with `#seq · iter N · E … Eh` from the latest checkpoint), then settles
  **converged** (green) or **failed** (red) on the terminal snapshot. When
  the job finishes, the status bar announces *"Job converged, final
  results loaded"*.
- **Streaming sections.** A section the producer is still growing (an
  optimization `trajectory`, SCF history mid-run) is badged *streaming* in
  the sidebar and reloads in place as it grows.

You don't have to fetch the finished job separately: when `run_status`
leaves `running`, the checkpoint you are already watching *is* the settled
result.

## Python API

The same machinery is importable for scripting:

```python
from vibeview.file_watcher import QVFChangeTracker, watch_qvf

# Synchronous: poll from your own loop.
tracker = QVFChangeTracker("job.qvf")          # settle_delay=0.5
event = tracker.poll()                         # QVFChange | None
if event:
    print(event.changed_sections, event.added_sections)

# Or a background thread with callbacks:
watcher = watch_qvf("job.qvf", on_reload, interval=5.0,
                    on_event=lambda ev: print(ev.touched_sections))
watcher.stop()
```

`QVFChange` carries `changed_sections` / `added_sections` /
`removed_sections` (manifest section ids), `manifest_meta_changed` (True
when non-section manifest keys moved, for example a streaming job's
`provenance.run_status` flipping to `converged`), and `is_qvf` (False for
a file tracked by whole-file hash because it isn't a parseable QVF
archive).

The streaming provenance is on the reader itself:

```python
from vibeview.qvf import QVFReader

r = QVFReader("checkpoint.qvf")
r.run_status                 # "running" | "converged" | "failed" | None
r.checkpoint_info            # {"seq": int, "wall_time_s": float, ...}
r.section_is_partial("traj0")  # True while the producer is still growing it
```

## See also

- [vq Job Manager](vibe_view_vq_jobs.md), where **Watch live** lives.
- [QVF consumer reference](../consumer_qvf_reference.md), the producer-side
  contract for streaming checkpoints.
- [vibe-view: interactive viewer](vibe_view.md), the rest of the viewer.
