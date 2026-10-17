/** Independently installed tools. Availability is explicit: do not invent public
 * source, release or manual links for a product that has not published them.
 * QVF is a format contract, not a fifth application.
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
    "installation": "Clone the public core repository, then follow the installation guide for your platform prerequisites. The installer builds native dependencies and creates a dedicated Python environment. The commands below select main; without --dev, the installer selects the release branch or the newest stable tag when that branch is absent.",
    "installGuide": "/docs/installation.html",
    "guides": [
      ["Tutorial learning paths", "/docs/tutorial/index.html#choose-a-learning-path"],
      ["Feature matrix", "/docs/features.html"],
      ["Validation", "/docs/validation.html"]
    ],
    "command": "git clone https://github.com/vibe-qc/vibe-qc.git\ncd vibe-qc\n./scripts/install.sh --dev",
    "capabilities": [
      [
        "Molecular methods",
        "Hartree-Fock, DFT with over 500 functionals, MP2, coupled cluster through CCSD(T) and DLPNO-CCSD(T), TDDFT, CASSCF with CASPT2 and NEVPT2, and determinant solvers up to full CI."
      ],
      [
        "Periodic systems",
        "One-, two-, and three-dimensional HF and KS-DFT on the GDF, GPW and BIPOLE Coulomb routes, k-point sampling, smearing for metals, band structures, densities of states and DFT+U."
      ],
      [
        "Fast surfaces",
        "MSINDO, DFTB, GFN2-xTB and PM6 semiempirical Hamiltonians and the optional MACE potential for screening and preoptimisation, each with its own status label."
      ],
      [
        "Properties and workflows",
        "Analytic gradients and Hessians, frequencies and thermochemistry, optimisation, NEB, IRC, dynamics, implicit solvation, population and bonding analysis, QTAIM and COOP/COHP."
      ],
      [
        "Reusable results",
        "Write QVF archives for independent visualization. The separately installable vibe-basis toolkit remains in this repository."
      ],
      [
        "Labelled status",
        "Every route is labelled production, preview, experimental or gated, and an experimental route warns at runtime. The cyclic cluster model lines are research output, not validated methods."
      ]
    ],
    "independence": "Install the engine for calculations. Add the viewer or queue when your workflow needs them; each has its own environment and releases."
  },
  {
    "id": "vibe-view",
    "name": "vibe-view",
    "action": "Analyze",
    "headline": "See what your calculation found.",
    "summary": "Explore structures, orbitals, densities, and spectra from QVF archives and supported chemistry file formats.",
    "audience": "For researchers inspecting results, collaborators opening a shared calculation, and anyone preparing scientific figures.",
    "docs": "/vibe-view/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-view",
    "requirements": "Python 3.11 or newer and Git. The source installer supports macOS and Linux; no vibe-qc native build is required.",
    "installation": "Clone the separate public vibe-view repository and run its installer from that checkout. The installer creates its own .venv and installs the browser and terminal interfaces. Desktop packaging and platform availability are documented by the viewer project.",
    "installGuide": "/vibe-view/docs/installation.html",
    "guides": [
      ["Quickstart", "/vibe-view/docs/quickstart.html"],
      ["CLI reference", "/vibe-view/docs/cli.html"]
    ],
    "command": "git clone https://github.com/vibe-qc/vibe-view.git\ncd vibe-view\n./scripts/install.sh\n.venv/bin/vibe-view --help",
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
    "action": "Submit",
    "headline": "Keep work moving across machines.",
    "summary": "Queue, dispatch, and track command-line workloads on local machines, remote hosts, and configured scheduler backends.",
    "audience": "For researchers managing long calculations and teams sharing workstations or compute hosts.",
    "docs": "/vibe-queue/docs/",
    "docsPending": false,
    "repository": "https://github.com/vibe-qc/vibe-queue",
    "requirements": "Python 3.12 or newer and Git. SSH access and any external scheduler setup are needed only for the hosts you use.",
    "installation": "Clone the separate public vibe-queue repository and run its installer from that checkout. The default installer makes a dedicated .venv with the CLI and daemon. Configure hosts and services using the queue guide after installation.",
    "installGuide": "/vibe-queue/docs/lifecycle.html#source-install-lifecycle",
    "guides": [
      ["Running jobs", "/vibe-queue/docs/user/index.html"],
      ["Running a host", "/vibe-queue/docs/operator/index.html"],
      ["The agent contract", "/vibe-queue/docs/agent/index.html"]
    ],
    "command": "git clone https://github.com/vibe-qc/vibe-queue.git\ncd vibe-queue\n./scripts/install.sh\n.venv/bin/vq --help",
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
    "independence": "The repository and product are vibe-queue. The shell command and import package remain vq; the Python distribution is vibe-queue. Installing the queue does not build or install vibe-qc."
  },
  {
    "id": "vibe-sketch",
    "name": "vibe-sketch",
    "action": "Sketch",
    "headline": "From a chemical idea to a calculation input.",
    "summary": "Draw structures, reactions and mechanisms. Prepare 3D geometries, calculation inputs and publication graphics.",
    "audience": "For chemists preparing molecular structures, reaction schemes and inputs for electronic-structure calculations.",
    "docs": "/vibe-sketch/docs/",
    "docsPending": false,
    "repository": null,
    "status": "Testing version",
    "license": "Proprietary",
    "requirements": "Python 3.12 or newer. Node.js 22 or newer to build the browser interface. The source installer supports macOS and Linux.",
    "installation": "Access is currently arranged with the maintainer. From an authorized checkout, the installer creates a dedicated environment and builds the browser app when npm is available. The command below opens your local installation; this website does not run a shared sketching service.",
    "installGuide": "/vibe-sketch/docs/installation.html",
    "guides": [
      ["Quickstart", "/vibe-sketch/docs/quickstart.html"],
      ["Calculation inputs", "/vibe-sketch/docs/inputs.html"],
      ["Python API", "/vibe-sketch/docs/python_api.html"]
    ],
    "command": "./scripts/install.sh\n.venv/bin/vibe-sketch serve --open",
    "capabilities": [
      [
        "Structures and reactions",
        "Draw or import molecular structures, reactions and mechanisms, clean up the layout, and export SVG or PNG figures. Exchange structures through MOL, SDF, RXN, SMILES and InChI."
      ],
      [
        "Calculation preparation",
        "Generate 3D geometries and write inputs for vibe-qc and ORCA. Writers reject job settings they cannot represent instead of silently substituting a method."
      ],
      [
        "Automation interfaces",
        "Use the Python SDK, command line, REST API or MCP server for supported drawing, conversion and calculation-preparation operations."
      ],
      [
        "Optional connections",
        "Checked names use the vibe-qc naming engine. QVF document saving needs the optional qvf-writer integration; without it, documents save as .vsk. Viewer and queue connections require their own installations."
      ],
      [
        "Availability",
        "The browser workflow is a testing version with a public manual. Public source and qualified desktop downloads are not published here yet. Request software access before following the checkout instructions."
      ],
      [
        "Licensing",
        "vibe-sketch is currently proprietary. Its license is separate from the MPL-2.0 engine, viewer and queue and the Apache-2.0 QVF project."
      ]
    ],
    "independence": "vibe-sketch has its own installation and version line. Drawing and exporting structures does not require a native vibe-qc build. Optional integrations are configured separately."
  }
]);
