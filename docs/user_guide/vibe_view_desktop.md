# vibe-view desktop app (macOS, Windows, Linux)

Source commands on this page run from the **separate vibe-view checkout**.
Clone [mpei/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](../tutorial/vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

The vibe-view source checkout includes a native desktop application (an
Electron window) in addition to the browser-based viewer. It gives you a real
app window with native menus, drag-and-drop, an Open Recent list, and (on
macOS) a double-click file association for `.qvf` files. The same codebase has
targets for macOS, Windows, and Linux.

## Choose the desktop route

| Route | Best for | Python environment |
|---|---|---|
| `vibe-view desktop` from a source checkout | current supported development and user route | created by `scripts/install.sh` |
| Packaged `.app`, `.dmg`, `.deb`, AppImage, or NSIS build | native menus and file association without a source tree | requires a qualified package and its bundled or discovered runtime |

For a first installation, follow the viewer's
[installation guide](https://vibe-qc.com/vibe-view/docs/installation.html) and
[quickstart](https://vibe-qc.com/vibe-view/docs/quickstart.html). Those pages
track its current source-install and desktop requirements. This core page
provides integration context and packaging design details.

```{note}
The desktop app is a thin native shell around vibe-view's Python server:
it launches `vibe-view serve` in the background and shows it in a window.
So it needs **Python and vibe-view installed** on the machine (see
Prerequisites).

If you launch the app and it can't find a suitable Python, it no longer
fails with a dead-end error; it shows a **setup screen** that names exactly
what is missing (Python, vibe-view, or the `[viewer]` extra) and lets you press
**recheck** after installing it (see [First run](#first-run-setup-screen)
below). Until the distribution is published, the screen installs the
same-version wheel URL recorded by that viewer release. This legacy
bootstrap URL is a companion release-maintenance concern after the split;
use the source setup below until its artifact has been verified.

**What is available today:** desktop mode runs from a source checkout. Python
wheels contain browser, terminal, and headless features. Wheels do not contain
the Electron source. Old viewer downloads under the core documentation are
legacy monorepo artifacts; use the viewer's own installation and release
information to choose a current artifact. Signed, notarized standalone desktop artifacts have not
landed yet. An older ad-hoc-signed macOS arm64 2.10.0 package remains on the
update feed for testing; it can detect an update and open a replacement DMG,
but it cannot self-install and is older than the supported source route.
Locally built desktop packages are also **not** a self-contained Python
bundle: they need the `vibeview` Python distribution with its `[viewer]`
extra. Bundling a full Python + VTK runtime into the installer was
evaluated and deferred (see
[`docs/desktop_packaging_design.md`](https://github.com/vibe-qc/vibe-view/blob/main/docs/desktop_packaging_design.md)).
```

## Prerequisites (all platforms)

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 or newer | https://python.org/ |
| vibe-view | current | installed with the `[viewer]` extra (below) |

The product and command are `vibe-view`; the Python distribution and import
package are `vibeview`. The distribution is not on PyPI yet, so install it
from its separate source checkout or from a validated artifact on its own
release page. Legacy wheels in the core docs downloads are not current releases.

## Install

On macOS and Linux, the standalone installer creates a dedicated viewer
environment with the browser server and records it for direct app launches:

```sh
./scripts/install.sh
source .venv/bin/activate
```

The lifecycle shell scripts target macOS and Linux. On Windows, create the
environment explicitly from PowerShell in the checkout root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[viewer]"
.venv\Scripts\vibe-view.exe --version
.venv\Scripts\vibe-view.exe desktop
```

Repeat the `python.exe -m pip install -e` command after updating the source
checkout. The macOS Applications-copy management described below does not
apply on Windows.

Electron downloads automatically on the first `vibe-view desktop` launch. To
download it during the initial setup instead, run
`./scripts/install.sh --with-electron`. For an existing standalone
environment, quit the desktop window and update the checkout, Python package,
and source-backed desktop application together:

```sh
./scripts/update-desktop.sh
```

The updater validates the existing Electron engine and synchronizes a missing,
corrupt, or mismatched engine to the reviewed version in `package-lock.json`.
On macOS it also refreshes the Applications copy and its displayed vibe-view
version. It records which checkout owns that source app and refuses to take
over one from another checkout unless you pass `--adopt-desktop`. Inspect the
reported path before using that option. It does not replace separately built
or future downloaded packages. On Linux the updater synchronizes Electron and
its launch wrappers, then the window continues to start through
`vibe-view desktop`; there is no Applications copy.

Run install, update, reinstall, and uninstall as your regular login user,
without `sudo`. Interactive privileged execution is refused so it cannot leave
a root-owned environment that the desktop session cannot maintain. The release
test runner has one narrow compatibility path that requires both `CI=true` and
`GITLAB_CI=true`; a sudo environment is still refused, and Linux desktop launch
always refuses root.

For a combined environment, use [Install both](../getting_started.md#install-both)
to install from the independent viewer checkout. The core repository no longer
has a local-source mapping for vibe-view.

The `[viewer]` extra is required for the desktop app: it pulls the
interactive server stack (trame, uvicorn). Installing vibe-view without it
gives the lean headless-capture stack only, and the desktop window will
report that the server did not start.

Use `./scripts/update.sh` when only the checkout and standalone
Python environment need updating. Use
`./scripts/update-desktop.sh` when the source-backed desktop shell
should be refreshed too.

For a clean repair from the current checkout, without fetching Git, quit every
desktop window and running CLI/browser-server launch, then run:

```sh
./scripts/reinstall.sh --desktop
```

To remove a source installation, quit every desktop window and CLI launch, run
`uninstall.sh --dry-run` first, and then run `uninstall.sh`. The uninstaller
removes only this checkout's recognizable Electron runtime, macOS source
app, and the Dock tile / command link created by `install.sh --dock` /
`--link-bin`; packaged apps and apps owned by another checkout are left
untouched. It
also removes the selected standalone viewer
environment. Settings, recent files, logs, window state, QVF files, and the
first-run app-managed environment at `$XDG_DATA_HOME/vibe-view/venv` are
preserved. Pass `--keep-desktop` to keep the Electron runtime and source app.

Verify:

```sh
vibe-view --version
# vibe-view 2.15.2  -- Roothaan's Roadrunner   (your version may differ)
```

## Launch

```sh
vibe-view desktop                 # native window, browse page for the cwd
vibe-view desktop water.qvf       # native window with water.qvf loaded
vibe-view desktop --port 9090     # custom server port
```

`vibe-view desktop` works from any directory. On first run it downloads the
reviewed Electron binary (about 120 MB) and brands it as vibe-view. It
then records the interpreter it ran from, so a later double-click on a
`.qvf` file (macOS) reuses the same virtualenv.

### Dock icon and shell command

Two installer integrations give the source install an app-store feel on
macOS. To pin the source-backed `vibe-view.app` to the Dock and expose the
`vibe-view` command in every shell:

```sh
./scripts/install.sh --with-electron --dock \
    --link-bin /opt/homebrew/bin
```

`--dock` appends the app to the Dock's persistent tiles and asks the Dock
to reload; running it again is a no-op. `--link-bin DIR` writes a marked
`vibe-view` launcher into an existing writable directory on your PATH
(Homebrew macOS: `/opt/homebrew/bin`; `~/.local/bin` when it is on PATH)
and refuses to overwrite a file it did not create. `--adopt-desktop`
transfers the app bundle from another checkout. `uninstall.sh` removes the
command link with the environment and unpins the Dock tile with the app,
and re-running `install.sh` with the same flags refreshes both after the
checkout moves.

(first-run-setup-screen)=

## First run: setup screen

When the app starts it looks for a Python that can run the viewer server. If
it can't find one (for example, a locally built app on a machine with no
vibe-view, or a Python that has vibe-view but not the `[viewer]` extra), it opens a
**setup screen** instead of failing. The screen tells you which of three
cases applies and shows a command to fix it.

The setup screen's legacy wheel URL must be migrated by the viewer release
owner before it can serve as a current installation route. In the meantime,
use an authorized source checkout and a dedicated environment:

```sh
python3 -m venv ~/.local/share/vibe-view/venv
```

From an authorized source checkout, the final line may instead be:

```sh
~/.local/share/vibe-view/venv/bin/pip install -e "$PWD[viewer]"
```

Download the current wheel from the
[viewer-only installation tutorial](../tutorial/vibe_view_getting_started.md).
When vibe-view is published on PyPI, the package-name form
`python -m pip install "vibeview[viewer]"` will become equivalent.

```{admonition} Why a virtualenv, not a plain `pip install`?
:class: important
On Homebrew Python (macOS) and Debian / Ubuntu / Fedora system Pythons, a
plain system-level pip install fails with **`error:
externally-managed-environment`** ([PEP 668](https://peps.python.org/pep-0668/)):
those interpreters are marked off-limits to `pip`. A **fresh virtualenv is
never externally managed**, so installing into one sidesteps the error
entirely. The app looks for this venv at
`$XDG_DATA_HOME/vibe-view/venv` (default `~/.local/share/vibe-view/venv`;
`%LOCALAPPDATA%\vibe-view\venv` on Windows), so **Recheck finds it
automatically** with no extra step. Installing into the system Python with
`--break-system-packages` is possible but discouraged.
```

Run the checkout command above in a terminal, then press the **recheck**
button. The app re-scans and, once it finds a capable
interpreter, starts the server and opens the viewer, with no restart needed.
The app stays open the whole time, so you can fix the environment without
losing your place.

```{note}
Bare `pip install "vibeview[viewer]"`, `pipx install vibeview`, and public-index
`uv tool install` commands do not resolve until the distribution is published.
Use the separate viewer checkout path shown above. The virtualenv is
what avoids the PEP 668 error. Launching with `vibe-view desktop` from that
environment records the interpreter, so the setup screen is skipped on later
runs.
```

## macOS: double-click a `.qvf`

Two ways to open files:

1. **`vibe-view desktop <file>`** works immediately after install, no app
   bundle needed.
2. **Double-click a `.qvf`** (Finder file association) needs the packaged
   `.app`. Build it once from the checkout:

   ```sh
   cd electron
   ./build-desktop.sh          # produces dist/mac*/vibe-view.app + a .dmg
   ```

   electron-builder uses `dist/mac-arm64/` on Apple silicon and `dist/mac/`
   on Intel Macs.

   This creates a local development artifact, not a signed downloadable
   release. Then run `vibe-view desktop` once from your virtualenv so the app knows
   which Python to use. After that, double-clicking a `.qvf` opens it in the
   app.

The branded app declares every format it can read: `.qvf`, `.py`
(vibe-qc inputs), and the loose structure/volume formats (`xyz`, `cif`,
`cube`, `pdb`, `mol2`, `sdf`, `mol`, `gjf`, `com`, `gro`), so **right-click
→ Open With → vibe-view** works for all of them from Finder, and a `.py`
input opens with its structure loaded (the parser reads the input
library's `build_system` / `build_molecule` style without executing the
file). The handler rank is Alternate: vibe-view never steals the default
Python-file association. After reinstalling or adopting the app,
macOS usually picks up the declarations on first launch; to force an
immediate refresh:

```sh
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f /Applications/vibe-view.app
```

```{note}
The locally built `.app` / `.dmg` is ad-hoc signed, but it is **not Developer
ID-signed or notarized**. macOS Gatekeeper may therefore block its first
launch. Right-click the app and choose **Open** once to allow it. Developer ID
signing and notarization remain a downloadable-packages roadmap item.
```

## Windows and Linux

`vibe-view desktop` from a source checkout is the desktop route on Windows and
Linux too, with the same prerequisites (Python plus the `[viewer]` extra).
electron-builder targets exist for Windows (`nsis` installer) and Linux
(`AppImage`), but no signed or supported download artifacts are published.
Locally built packages rely on a Python environment being present, and the
first-run [setup screen](#first-run-setup-screen) reports what is missing.

On Linux, install and launch vibe-view as your regular login user. Do not run
`vibe-view desktop` with `sudo`: Electron's security sandbox does not support a
root desktop session, and vibe-view will refuse that unsafe launch explicitly.
Linux also needs Electron's normal GTK3 desktop runtime. Desktop installations
usually provide it already. For a minimal Debian 12 or Ubuntu 22.04 system:

```sh
sudo apt-get install libgtk-3-0 libnss3 libatk-bridge2.0-0 \
  libdrm2 libxkbcommon0 libgbm1 libasound2
```

Debian 13 and Ubuntu 24.04 use time64 package names:

```sh
sudo apt-get install libgtk-3-0t64 libnss3 libatk-bridge2.0-0t64 \
  libdrm2 libxkbcommon0 libgbm1 libasound2t64
```

If a container or host security policy blocks Chromium user namespaces,
configure a supported sandbox policy when possible. For a trusted local file,
`vibe-view desktop --no-sandbox` is an explicit last-resort workaround; it
prints a warning because disabling the sandbox reduces isolation.

On Windows the app looks for the `py` launcher and `python` (in that order)
rather than `python3`, which Windows usually doesn't provide; a virtualenv is
found at `.venv\Scripts\python.exe`.

```{warning}
**The Windows target is not yet verified, and we want developer feedback.** The
maintainers develop and test on macOS and Linux and have no Windows machine,
so the locally built `nsis` installer, the
`py`-launcher Python discovery, and the setup screen have not been run on real
Windows. The app says as much in a one-time notice on its first Windows
launch.

Because there is no Windows machine on this side, Windows problems cannot be
tested or reproduced directly here -- but **if you want to try it, you will get
help.** The developer is glad to support anyone running vibe-view on Windows
and to work through whatever you hit. Please tell us how it goes, whether it
works or breaks, by
[opening an issue](https://github.com/vibe-qc/vibe-qc/issues); that
feedback is what will make Windows a first-class target. No Windows download
artifact is published.
```

## Troubleshooting

**The setup screen keeps appearing after I installed vibe-view.** The app
scans a fixed set of interpreters (`python3`, `python`, `python3.13` … on
macOS/Linux; `py`, `python` on Windows) plus the interpreter recorded by a
previous `vibe-view desktop` run. If you installed the `vibeview` distribution
with its `[viewer]` extra into a virtualenv that isn't on that list, the
app won't find it from a bare double-click. Run `vibe-view desktop` **once**
from that environment; it records the interpreter (in
`<XDG_CACHE_HOME>/vibe-view/interpreter.json`) so future launches and
double-clicks reuse it. Confirm the environment is capable with:

```sh
python -c "import vibeview, trame, uvicorn; print('ok')"
```

**"The vibe-view server did not start."** Older builds showed this error and
quit; current builds route the same conditions to the
[setup screen](#first-run-setup-screen) instead, which names what's missing
and lets you fix it in place.

**"Check for Updates".** The hosted macOS arm64 feed currently advertises the
older 2.10.0 ad-hoc package. It can detect and offer a manual DMG download, but
Developer ID signing is still required for self-install, and the feed never
updates a source checkout. Linux and Windows feeds are not published. For the
supported source route, quit the app and run
`./scripts/update-desktop.sh` from the checkout root. Rebuild
Electron only when developing or testing a native artifact. Feed, signing,
and publishing details are in
`electron/PUBLISHING.md`.

## See also

- [vibe-view interactive viewer](vibe_view.md) (the browser-based viewer
  and the full section-by-section reference)
- `electron/PUBLISHING.md` (building, the update feed, and the
  code-signing gate for auto-update)
- `electron/README.md` (developer notes for the desktop app)
