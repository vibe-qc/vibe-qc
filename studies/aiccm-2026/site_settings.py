"""External site configuration for study launchers and legacy record readers.

Scientific inputs stay in this directory. Select a private JSON profile outside
the source checkout with AICCM_SITE_CONFIG. No profile means no default remote
submission target; an explicit CLI target remains available.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket

_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_MAPS = {"tier_hosts", "crystal_tier_hosts", "coverage_hosts", "system_hosts"}
_HOSTS = {"paper_host", "posthf_host", "comparison_host"}
_LISTS = {"molecular_hosts", "local_host_aliases"}
_KEYS = _MAPS | _HOSTS | _LISTS | {
    "schema_version", "allow_submission", "bundle_attestation_mode",
}
PUBLIC_BUNDLE_MODE = "aiccm-bundle-attestation/v1"


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate site-profile key")
        result[key] = value
    return result


def validate_profile(profile: dict) -> dict:
    if not isinstance(profile, dict) or type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError("site profile requires schema_version 1")
    if set(profile) - _KEYS:
        raise ValueError("site profile contains an unknown setting")
    if type(profile.get("allow_submission", False)) is not bool:
        raise ValueError("allow_submission must be a boolean")
    for key in _MAPS:
        if key not in profile:
            continue
        if not isinstance(profile[key], dict):
            raise ValueError("site host map must be an object")
        for name, value in profile[key].items():
            if not isinstance(name, str) or not isinstance(value, str) or not value:
                raise ValueError("site host map requires nonempty strings")
            # Historic system labels can name alternate hosts. They are metadata,
            # never shell arguments or submission targets.
            if key == "system_hosts":
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", value):
                    raise ValueError("invalid historical system-host label")
            elif not _HOST.fullmatch(value):
                raise ValueError("invalid site host alias")
    for key in _HOSTS:
        if key in profile and (not isinstance(profile[key], str) or not _HOST.fullmatch(profile[key])):
            raise ValueError("invalid site host alias")
    for key in _LISTS:
        if key in profile and (not isinstance(profile[key], list) or not profile[key]
                or any(not isinstance(v, str) or not _HOST.fullmatch(v) for v in profile[key])):
            raise ValueError("site host list requires nonempty host aliases")
    mode = profile.get("bundle_attestation_mode")
    if mode is not None and (not isinstance(mode, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", mode)):
        raise ValueError("invalid legacy attestation mode")
    return profile


def load_profile() -> dict:
    selected = os.environ.get("AICCM_SITE_CONFIG")
    if selected is None:
        return {}
    if not selected or not Path(selected).is_absolute():
        raise ValueError("AICCM_SITE_CONFIG must select an external private file")
    root = Path(__file__).resolve().parent
    for ancestor in root.parents:
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "python/vibeqc").is_dir():
            root = ancestor
            break
    try:
        path = Path(selected).expanduser().resolve(strict=True)
        if path.is_relative_to(root):
            raise ValueError("site configuration must be outside the source checkout")
        # Check both the configured interface and resolved target, rejecting a checkout, worktree
        # or bare Git database as well as this source tree. Private profiles
        # must not become source or accidentally enter Git object storage.
        for interface in (Path(selected).absolute(), path):
            if interface.is_relative_to(root.resolve()):
                raise ValueError("private configuration must stay outside the source checkout")
            for directory in interface.parents:
                if ((directory / ".git").exists()
                        or (directory / ".git").is_symlink()
                        or ((directory / "HEAD").is_file()
                            and (directory / "objects").is_dir()
                            and (directory / "refs").is_dir())
                        or directory.name.casefold() == ".git"):
                    raise ValueError("private configuration must stay outside Git trees and databases")
        with path.open("rb") as stream:
            data = stream.read(65537)
        if len(data) > 65536:
            raise ValueError("site profile exceeds the size limit")
        return validate_profile(json.loads(data, object_pairs_hook=_pairs))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("cannot read the external site profile; path redacted") from None


_profile = load_profile()


def profile_id() -> str | None:
    if not _profile:
        return None
    return "sha256:" + hashlib.sha256(json.dumps(_profile, sort_keys=True).encode()).hexdigest()


def local_hosts() -> set[str]:
    name = socket.gethostname().casefold().rstrip(".")
    short = name.split(".")[0]
    return {"localhost", "localhost.localdomain", "local", "127.0.0.1", "::1",
            name, short, short + ".local"} | {
        value.casefold().rstrip(".") for value in _profile.get("local_host_aliases", [])
    }


def is_local_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.casefold().rstrip(".") in local_hosts():
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def explicit_remote_host(value: str) -> str:
    if not isinstance(value, str) or not _HOST.fullmatch(value) or is_local_host(value):
        raise ValueError("select an explicit non-local queue host alias")
    return value


def require_submission() -> None:
    if _profile.get("allow_submission") is not True:
        raise ValueError("remote defaults require an explicitly enabled external site profile")


def required_host(key: str, name: str | None = None) -> str:
    require_submission()
    value = _profile.get(key)
    if name is not None:
        value = value.get(name) if isinstance(value, dict) else None
    if value is None:
        raise ValueError("required remote target is missing from the external site profile")
    return explicit_remote_host(value)


class _HostMap(Mapping):
    def __init__(self, key):
        self.key = key

    def __getitem__(self, name):
        return required_host(self.key, name)

    def __iter__(self):
        return iter(_profile.get(self.key, {}))

    def __len__(self):
        return len(_profile.get(self.key, {}))


def host_map(key: str) -> Mapping:
    return _HostMap(key)


def molecular_hosts() -> tuple[str, ...]:
    require_submission()
    values = _profile.get("molecular_hosts")
    if not values:
        raise ValueError("molecular host rotation is missing from the external site profile")
    return tuple(explicit_remote_host(value) for value in values)


def system_host(name: str, default: str | None = None) -> str | None:
    return _profile.get("system_hosts", {}).get(name, default)


def comparison_host() -> str | None:
    return _profile.get("comparison_host")


def bundle_mode() -> str:
    return _profile.get("bundle_attestation_mode", PUBLIC_BUNDLE_MODE)


def accepted_bundle_modes() -> tuple[str, ...]:
    return tuple(dict.fromkeys((PUBLIC_BUNDLE_MODE, bundle_mode())))
