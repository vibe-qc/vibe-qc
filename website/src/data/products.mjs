/** Product ownership is static; companion versions stay in their repositories.
 *
 * Membership rule: an entry here is a tool a researcher installs and runs to do
 * work. Each one generates a page at `/products/<id>/` and a card on both the
 * homepage and get-started. The object shape encodes that rule, because
 * `requirements`, `installation`, `installGuide` and `command` only have honest
 * answers for something installable.
 *
 * QVF (project 37) is deliberately not a member, and its absence is a decision
 * rather than an omission. QVF is a format specification with an Apache-2.0
 * reference toolkit that nobody installs in order to use this suite: vibe-qc
 * and vibe-view each implement the format independently and validate against
 * the published contract, and `docs/roadmap.md` records that nothing depends on
 * it at runtime or install time. Two implementations, one spec. Listing it here
 * would invent an install story that does not exist and would undercut the
 * independence the separate project exists to demonstrate.
 *
 * The site carries QVF in the tiers where it does belong: the "Four
 * repositories" grid on get-started, the suite section on the homepage, and the
 * format block on every product page. A fourth entry here would also falsify
 * the hero eyebrow in `pages/index.astro` and the description in
 * `layouts/Base.astro`, which count products rather than repositories.
 * `tests/products.verify.mjs` guards this split.
 */
export const products = Object.freeze([
  {
    "id": "vibe-qc",
    "name": "vibe-qc",
    "action": "Calculate",
    "headline": "From a molecule to a crystal.",
    "summary": "Electronic-structure calculations for molecules and periodic solids, with a Python front end and a C++17 engine.",
    "audience": "For researchers calculating energies, structures, spectra, and electronic properties of molecules and materials.",
    "docs": "/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-qc",
    "requirements": "Python 3.11 or newer, a C++17 toolchain, and the native libraries described in the installation guide.",
    "installation": "Clone the core repository using the access instructions in GitLab, then follow the installation guide for your platform. The installer builds native dependencies and creates a dedicated Python environment. The default installer selects the release branch; --dev selects main.",
    "installGuide": "/docs/installation.html",
    "guides": [],
    "command": null,
    "capabilities": [
      [
        "Molecular methods",
        "Hartree-Fock, density-functional theory, MP2, coupled cluster, and multireference methods."
      ],
      [
        "Periodic systems",
        "One-, two-, and three-dimensional HF and KS-DFT, k-point sampling, band structures, and densities of states."
      ],
      [
        "Reusable results",
        "Write QVF archives for independent visualization. The separately installable vibe-basis toolkit remains in this repository."
      ]
    ],
    "independence": "Install the engine for calculations. Add the viewer or queue when your workflow needs them; each has its own environment and releases."
  },
  {
    "id": "vibe-view",
    "name": "vibe-view",
    "action": "Explore",
    "headline": "See what your calculation found.",
    "summary": "Explore structures, orbitals, densities, and spectra from QVF archives and supported chemistry file formats.",
    "audience": "For researchers inspecting results, collaborators opening a shared calculation, and anyone preparing scientific figures.",
    "docs": "/vibe-view/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-view",
    "requirements": "Python 3.11 or newer and Git. The source installer supports macOS and Linux; no vibe-qc native build is required.",
    "installation": "Clone the separate vibe-view repository using GitLab, then run these commands from that checkout. The installer creates its own .venv and installs the browser and terminal interfaces. Desktop packaging and platform availability are documented by the viewer project.",
    "installGuide": "/vibe-view/docs/installation.html",
    "guides": [
      ["Quickstart", "/vibe-view/docs/quickstart.html"],
      ["CLI reference", "/vibe-view/docs/cli.html"]
    ],
    "command": "./scripts/install.sh\n.venv/bin/vibe-view --help",
    "capabilities": [
      [
        "Choose your interface",
        "Use the browser, a desktop window, or an interactive terminal to inspect a result."
      ],
      [
        "Make a figure",
        "Capture views headlessly or work with results from Python for repeatable figure preparation."
      ],
      [
        "Bring your own producer",
        "Read QVF archives from vibe-qc or another code that implements the format, plus supported structure and volume files."
      ]
    ],
    "independence": "vibe-qc is not required to install or use the viewer. The command is vibe-view; the Python distribution and import package are vibeview."
  },
  {
    "id": "vibe-queue",
    "name": "vibe-queue",
    "action": "Schedule",
    "headline": "Keep work moving across machines.",
    "summary": "Queue, dispatch, and track command-line workloads on local machines, remote hosts, and configured scheduler backends.",
    "audience": "For researchers managing long calculations and teams sharing workstations or compute hosts.",
    "docs": "/vibe-queue/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-queue",
    "requirements": "Python 3.12 or newer and Git. SSH access and any external scheduler setup are needed only for the hosts you use.",
    "installation": "Clone the separate vibe-queue repository using GitLab, then run these commands from that checkout. The default installer makes a dedicated .venv with the CLI and daemon. Configure hosts and services using the queue guide after installation.",
    "installGuide": "/vibe-queue/docs/lifecycle.html#source-install-lifecycle",
    "guides": [
      ["Running jobs", "/vibe-queue/docs/user/index.html"],
      ["Running a host", "/vibe-queue/docs/operator/index.html"],
      ["The agent contract", "/vibe-queue/docs/agent/index.html"]
    ],
    "command": "./scripts/install.sh\n.venv/bin/vq --help",
    "capabilities": [
      [
        "Schedule commands",
        "Run vibe-qc jobs or other command-line workloads with the same queue."
      ],
      [
        "Follow the work",
        "Inspect queued and running jobs, their workspaces, and retrieved results."
      ],
      [
        "Choose your deployment",
        "Start with a local queue and add remote hosts or the optional web dashboard as needed."
      ]
    ],
    "independence": "The repository and product are vibe-queue. The shell command, Python distribution, and import package remain vq. Installing the queue does not build or install vibe-qc."
  }
]);
