# Portable cluster builds

The product provides portable build and installation helpers. Scheduler queues,
accounts, module choices, SSH aliases and deployment commands belong to the
operator's private configuration and operations repository.

Follow [installation](installation.md) and [contributor setup](contributor_setup.md)
inside an allocation with the required compiler, memory and CPU resources.
Use a separate source checkout and environment for each installed runtime.
Keep build work off shared login nodes unless the cluster permits it.

For a cluster without compute-node network access, stage dependency sources
and Python wheels on an authorized connected host, then transfer them using
your site's approved process. The generic native dependency helpers expose
`--fetch-only`; check each helper's `--help` for its supported options.
Prepare a wheelhouse for the target Python, platform and CPU baseline. Do not
assume a wheel that works on a build node also works on an older login node.

Build with the staged sources and an explicitly configured offline package
index. Preserve the selected source revision and dependency checksums with
the runtime. Validate the installed import and affected test lanes before
selecting that runtime for jobs. Never update a running job's environment in
place.

Site-specific provisioning wrappers are maintained separately from product
archives. The queue's external configuration selects installed commands;
product installation does not install or update those private wrappers.
See the [queue operator guide](https://vibe-qc.com/vibe-queue/docs/operator/index.html)
for scheduler configuration and runtime verification contracts.
