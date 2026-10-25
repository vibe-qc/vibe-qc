"""Executed MLIP identity and replayable choices for output records.

Model metadata describes the result; only public runner choices belong in
job.spec.options. A saved calculation never grants an academic license to
the next user who opens its archive.
"""


def mlip_job_options(result) -> dict:
    """Return the executed model/runtime choices, without license consent."""
    info = getattr(result, "model_info", None)
    if info is None:
        return {}
    options = {"model": info.key}
    for field in ("device", "dtype"):
        value = getattr(result, field, None)
        if value is not None:
            options[field] = str(value)
    identity = getattr(result, "weight_identity", None)
    if identity:
        options["weight_sha256"] = identity["sha256"]
    return {"mlip_options": options}


def mlip_record_fields(result) -> dict:
    """The same model attribution for .system and QVF provenance."""
    options = mlip_job_options(result)
    if not options:
        return {}
    info = result.model_info
    return {"mlip": {
        **options["mlip_options"],
        "loader": info.loader,
        "loader_arg": info.loader_arg,
        "license": info.license,
        "training_data": info.training_data,
        "theory": info.theory,
        "citation": info.citation,
        "doi": info.doi,
        "energy_scale": "model-specific reference-shifted",
        **({"checkpoint": dict(result.weight_identity)}
           if getattr(result, "weight_identity", None) else {}),
    }}
