/** Independently installed tools. Availability is explicit: do not invent public
 * source, release or manual links for a product that has not published them.
 * QVF is a format contract, not a fifth application.
 */
export const products = Object.freeze([
  {
    "id": "vibe-qc",
    "desktopDownload": true,
    "name": "vibe-qc",
    "action": "Calculate",
    "headline": "From a molecule to a crystal.",
    "summary": "Electronic-structure calculations for molecules and periodic solids, with a Python front end and a C++17 engine.",
    "audience": "For researchers calculating energies, structures, spectra, and electronic properties of molecules and materials.",
    "docs": "/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-qc",
    "requirements": "Standalone downloads include Python, native libraries and basis/ECP data: macOS 27 or newer on Apple Silicon, or Linux x86-64. Source builds require Python 3.11 or newer and a C++17 toolchain. Optional engines have their own dependency limits.",
    "installation": "Use the standalone package for the current release, including naming, structure fetching and D3/D4. Optional machine-learning models and developer/reference tools are separate. For a source build, clone the public core repository and follow the installation guide. The installer builds native dependencies and creates a dedicated Python environment. The commands below select main; without --dev, the installer selects the release branch or the newest stable tag when that branch is absent.",
    "installGuide": "/docs/installation.html",
    "guides": [
      [
        "Tutorial learning paths",
        "/docs/tutorial/index.html#choose-a-learning-path"
      ],
      [
        "Feature matrix",
        "/docs/features.html"
      ],
      [
        "Validation",
        "/docs/validation.html"
      ]
    ],
    "command": "git clone https://github.com/vibe-qc/vibe-qc.git\ncd vibe-qc\n./scripts/install.sh --dev",
    "capabilities": [
      [
        "Molecular methods",
        "Hartree-Fock, DFT with over 500 functionals, MP2, coupled cluster through CCSD(T) and DLPNO-CCSD(T), TDDFT, CASSCF with CASPT2 and NEVPT2, and determinant solvers up to full CI."
      ],
      [
        "Periodic systems",
        "Periodic HF and KS-DFT with route-specific dimensionality and spin support, GDF, GPW and BIPOLE Coulomb builders, k-point sampling, smearing, bands, densities of states and DFT+U. Consult the method matrix for supported combinations."
      ],
      [
        "Fast surfaces",
        "MSINDO, DFTB, GFN2-xTB and PM6 semiempirical Hamiltonians and the optional MACE potential for screening and preoptimisation, each with its own status label."
      ],
      [
        "Properties and workflows",
        "Geometry optimization, NEB, IRC and dynamics on supported energy surfaces; analytic or finite-difference derivatives where implemented; frequencies, thermochemistry, solvation, QTAIM and COOP/COHP within each method’s documented scope."
      ],
      [
        "Reusable results",
        "Write QVF archives for independent visualization. The separately installable vibe-basis toolkit includes basis optimization recipes and saves results with provenance."
      ],
      [
        "Labelled status",
        "Method families carry production, preview, experimental or gated labels with route-specific limits. Runtime warnings and recorded status vary by route. AICCM/CCM and SECCM remain experimental."
      ]
    ],
    "independence": "Install the engine for calculations. Add the viewer or queue when your workflow needs them; each has its own environment and releases.",
    "license": "MPL-2.0",
    "licenseGuide": "/docs/license.html"
  },
  {
    "id": "vibe-view",
    "desktopDownload": true,
    "name": "vibe-view",
    "action": "Analyze",
    "headline": "See what your calculation found.",
    "summary": "Inspect and edit structures, explore orbitals and volumes, and analyze spectra, bands and excited states from QVF and supported chemistry formats.",
    "audience": "For researchers inspecting results, collaborators opening a shared calculation, and anyone preparing scientific figures.",
    "docs": "/vibe-view/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-view",
    "requirements": "For a source installation: Python 3.11 or newer and Git. The source installer supports macOS and Linux. Public Linux x86-64 and macOS Apple Silicon packages include Python and the viewer runtime; download the current release from the downloads page. Apple Silicon package tooling targets macOS 14 or newer.",
    "installation": "The commands below install the browser, terminal, headless and source-checkout desktop interfaces in a dedicated environment. Self-contained Linux and macOS builds include PyVista, VTK, Trame and RDKit; they need no separate Python for normal viewing or editing. Current macOS and Linux installers are available from the downloads page. Mac downloads are ad-hoc signed and not Apple-notarized.",
    "installGuide": "/vibe-view/docs/installation.html",
    "guides": [
      [
        "Quickstart",
        "/vibe-view/docs/quickstart.html"
      ],
      [
        "CLI reference",
        "/vibe-view/docs/cli.html"
      ],
      [
        "Desktop app",
        "/vibe-view/docs/desktop.html"
      ],
      [
        "Supported data",
        "/vibe-view/docs/capabilities.html"
      ]
    ],
    "command": "git clone https://github.com/vibe-qc/vibe-view.git\ncd vibe-view\n./scripts/install.sh\n.venv/bin/vibe-view --help",
    "capabilities": [
      [
        "Choose your interface",
        "Use the browser, an Electron desktop window, an interactive terminal, Python or headless capture."
      ],
      [
        "Inspect and edit structures",
        "Measure and align structures, build molecules or crystals, edit atoms with undo/redo, and choose among the structures in an archive."
      ],
      [
        "Explore fields and orbitals",
        "Use isosurfaces or direct volume rendering with editable opacity transfer functions. Evaluate supported GTO orbitals on demand; higher angular momentum and Bloch evaluation have documented limits."
      ],
      [
        "Analyze calculated data",
        "Inspect spectra, bands, densities of states, trajectories, QTAIM topology and excited-state tables where the archive supplies them. Unsupported sections are reported explicitly."
      ],
      [
        "Make repeatable figures",
        "Use publication presets and headless capture, and retain camera, structure and display choices in bookmarks, slides or sessions."
      ],
      [
        "Bring your own producer",
        "Open QVF and supported chemistry formats without vibe-qc. Strict QVF validation uses the optional Apache-2.0 reference toolkit; opening a file does not require that extra."
      ]
    ],
    "independence": "vibe-qc is not required to install or use the viewer. The command is vibe-view; the Python distribution and import package are vibeview.",
    "license": "MPL-2.0",
    "licenseGuide": "https://github.com/vibe-qc/vibe-view/blob/main/LICENSE"
  },
  {
    "id": "vibe-queue",
    "desktopDownload": true,
    "name": "vibe-queue",
    "action": "Submit",
    "headline": "Keep work moving across machines.",
    "summary": "Submit and track command-line jobs locally, over SSH or through an HTTPS gateway, with daemon, PBS and Slurm execution backends.",
    "audience": "For researchers managing long calculations and teams sharing workstations or compute hosts.",
    "docs": "/vibe-queue/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-queue",
    "requirements": "Standalone downloads include Python, the CLI, daemon and web-console dependencies for macOS 14 or newer on Apple Silicon and Linux x86-64. Source installation requires Python 3.12 or newer and Git. Direct remote hosts use SSH; configured HTTPS gateway clients use scoped credentials without fleet SSH access. Scheduler targets require their own PBS or Slurm setup.",
    "installation": "Download the current standalone release to use vq without installing Python. For a source installation, clone the public vibe-queue repository and run its installer. The default installer makes a dedicated .venv with the CLI and daemon. Configure hosts and services using the queue guide after installation.",
    "installGuide": "/vibe-queue/docs/lifecycle.html#source-install-lifecycle",
    "guides": [
      [
        "Running jobs",
        "/vibe-queue/docs/user/index.html"
      ],
      [
        "Running a host",
        "/vibe-queue/docs/operator/index.html"
      ],
      [
        "The agent contract",
        "/vibe-queue/docs/agent/index.html"
      ],
      [
        "HTTPS submission",
        "/vibe-queue/docs/gateway.html"
      ],
      [
        "Portable runtimes",
        "/vibe-queue/docs/portable_runtime_bundles.html"
      ]
    ],
    "command": "git clone https://github.com/vibe-qc/vibe-queue.git\ncd vibe-queue\n./scripts/install.sh\n.venv/bin/vq --help",
    "capabilities": [
      [
        "Schedule commands",
        "Run vibe-qc or other command-line programs through local daemons, SSH hosts, PBS or Slurm. Track dependencies, arrays, retries and resource limits."
      ],
      [
        "Submit over HTTPS",
        "The optional gateway supports resumable uploads, idempotent submission, logs, cancellation and verified result downloads. An outbound connector reaches a broker behind a firewall; a deployment still needs its own TLS and access setup."
      ],
      [
        "Follow jobs and campaigns",
        "Inspect queued and running jobs, collected results and resource usage in the CLI or optional fleet console. Artifact campaigns coordinate dependent work and transfers."
      ],
      [
        "Manage runtimes",
        "Versioned runtime slots and portable bundles record payload identity and support rollback. Bundled runtime generations can bind to daemon jobs; scheduler binding and product-specific portability have separate limits."
      ],
      [
        "Estimate and explain",
        "Resource-estimate receipts and placement explanations are implemented. These are evidence and planning tools; they do not automatically change queue admission or placement."
      ]
    ],
    "independence": "The repository and product are vibe-queue. The shell command and import package remain vq; the Python distribution is vibe-queue. Installing the queue does not build or install vibe-qc.",
    "license": "MPL-2.0",
    "licenseGuide": "https://github.com/vibe-qc/vibe-queue/blob/main/LICENSE"
  },
  {
    "id": "vibe-sketch",
    "desktopDownload": true,
    "name": "vibe-sketch",
    "action": "Sketch",
    "headline": "From a chemical idea to a calculation input.",
    "summary": "Draw structures, reactions and mechanisms. Prepare 3D geometries, calculation inputs and publication graphics.",
    "audience": "For chemists preparing molecular structures, reaction schemes and inputs for electronic-structure calculations.",
    "docs": "/vibe-sketch/docs/",
    "docsPending": false,
    "repository": null,
    "status": "Testing version",
    "license": "MPL-2.0",
    "requirements": "For a source installation: Python 3.12 or newer, plus Node.js 22 or newer when building the browser or desktop interface. The Linux x86-64 package profile bundles Python, RDKit, QVF support and the editor. Apple Silicon package tooling targets macOS 14 or newer.",
    "installation": "The current vibe-sketch release is available for macOS Apple Silicon and Linux x86-64 with matching source and checksums on the downloads page. The commands below apply to a source checkout. macOS downloads are ad-hoc signed and not Apple-notarized. Windows editions remain planned paid Store purchases.",
    "installGuide": "/vibe-sketch/docs/installation.html",
    "guides": [
      [
        "Quickstart",
        "/vibe-sketch/docs/quickstart.html"
      ],
      [
        "Calculation inputs",
        "/vibe-sketch/docs/inputs.html"
      ],
      [
        "Python API",
        "/vibe-sketch/docs/python_api.html"
      ],
      [
        "Desktop app",
        "/vibe-sketch/docs/desktop.html"
      ],
      [
        "File formats",
        "/vibe-sketch/docs/formats.html"
      ]
    ],
    "command": "./scripts/install.sh\n.venv/bin/vibe-sketch serve --open",
    "capabilities": [
      [
        "Structures and reactions",
        "Draw or import molecular structures, reactions and mechanisms. Export scene-preserving SVG with ACS, RSC or Wiley JCC presets; the browser can rasterize schemes to PNG. Exchange structures through MOL, SDF, RXN, SMILES and InChI; The bundled publication renderer also exports PNG and PDF; CDXML export remains planned."
      ],
      [
        "Calculation preparation",
        "Generate 3D geometries and write vibe-qc, ORCA, QCSchema or XYZ inputs. Writers reject job settings they cannot represent. Submit supported jobs through a separately installed vibe-queue and open structures in vibe-view."
      ],
      [
        "Automation interfaces",
        "Use the Python SDK, command line, REST API or MCP server for supported drawing, conversion and calculation-preparation operations."
      ],
      [
        "Optional connections",
        "QVF is the default document format when qvf-writer is installed, and QVF support is included in the Linux and macOS package profiles. Source installations can save .vsk without it. Import supported structures and reactions from other QVF producers while protecting the original calculation files. Checked names use the optional vibe-qc naming engine."
      ],
      [
        "Availability",
        "The manual and current macOS and Linux desktop releases are public. Windows Store editions will be available for purchase after Store publication."
      ],
      [
        "Licensing",
        "vibe-sketch uses MPL-2.0, like vibe-qc, vibe-view and vibe-queue. QVF is a separate Apache-2.0 project. Third-party components retain their own licenses and notices."
      ]
    ],
    "independence": "vibe-sketch has its own installation and version line. Drawing and exporting structures does not require a native vibe-qc build. Optional integrations are configured separately.",
    "licenseGuide": "/vibe-sketch/docs/license.html"
  }
]);
