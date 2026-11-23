"""Safe adapter for KiCad's authoritative CLI schematic-to-netlist export."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import ImporterError
from .kicad_xml import KiCadXMLImporter


class KiCadCLIImporter:
    """Use KiCad CLI to export native schematics; never infer wiring geometrically."""

    def __init__(self, executable: str | Path | None = None, timeout: float = 120.0):
        self.executable = str(executable) if executable else (shutil.which("kicad-cli") or "")
        self.timeout = float(timeout)
        self.xml_importer = KiCadXMLImporter()
        if self.timeout <= 0:
            raise ValueError("KiCad CLI timeout must be positive.")

    def available(self) -> bool:
        if not self.executable:
            return False
        candidate = Path(self.executable).expanduser()
        return candidate.is_file() if candidate.parent != Path(".") else shutil.which(self.executable) is not None

    def version(self) -> str | None:
        if not self.available():
            return None
        try:
            result = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True,
                timeout=min(self.timeout, 15.0), check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        output = (result.stdout or result.stderr).strip()
        return output.splitlines()[0] if output else None

    def import_file(
        self,
        schematic: str | Path,
        *,
        variant: str | None = None,
        output_path: str | Path | None = None,
        preserve_netlist: bool = False,
        ai_provenance: dict[str, Any] | None = None,
    ):
        source = Path(schematic).expanduser().resolve()
        if source.suffix.lower() != ".kicad_sch":
            raise ImporterError(f"KiCad CLI input must have .kicad_sch extension: {source}")
        if not source.is_file():
            raise ImporterError(f"KiCad schematic does not exist: {source}")
        if not self.available():
            raise ImporterError(
                "kicad-cli was not found. Install KiCad or supply its executable path. "
                "A pre-exported .kicad_net XML file can be imported without KiCad CLI."
            )

        version = self.version()
        keep = bool(preserve_netlist or output_path is not None)
        if keep:
            destination = Path(output_path).expanduser().resolve() if output_path else source.with_suffix(".kicad_net")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Export beside the destination, then atomically replace it only after
            # a successful and readable export. A stale old file can never mask a
            # failed CLI invocation.
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{destination.stem}.", suffix=".kicad_net.tmp", dir=destination.parent
            )
            os.close(fd)
            temporary = Path(temp_name)
            temporary.unlink(missing_ok=True)
            try:
                command_result = self._run_export(source, temporary, variant)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            xml_path = destination
        else:
            with tempfile.TemporaryDirectory(prefix="cirkreety-kicad-") as tmp:
                xml_path = Path(tmp) / f"{source.stem}.kicad_net"
                command_result = self._run_export(source, xml_path, variant)
                result = self.xml_importer.import_file(
                    xml_path, original_schematic=source, variant=variant,
                    kicad_version=version, ai_provenance=ai_provenance,
                )
                self._add_cli_metadata(result, source, version, variant, command_result, kept=False, path=None)
                return result

        result = self.xml_importer.import_file(
            xml_path, original_schematic=source, variant=variant,
            kicad_version=version, ai_provenance=ai_provenance,
        )
        self._add_cli_metadata(result, source, version, variant, command_result, kept=True, path=xml_path)
        return result

    @staticmethod
    def _add_cli_metadata(result, source, version, variant, command_result, *, kept: bool, path):
        try:
            original_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError:
            original_sha256 = None
        xml_sha256 = result.circuit.source_information.get("sha256")
        result.metadata.update({
            "format": "kicad_schematic",
            "path": str(source),
            "filename": source.name,
            "sha256": original_sha256,
            "netlist_sha256": xml_sha256,
            "original_schematic": str(source),
            "kicad_cli": command_result["executable"],
            "kicad_version": version,
            "variant": variant,
            "generated_netlist_preserved": kept,
            "generated_netlist_path": str(path) if path else None,
            "cli_command": command_result["command"],
            "cli_returncode": command_result["returncode"],
            "cli_stdout": command_result["stdout"],
            "cli_stderr": command_result["stderr"],
            "original_schematic_sha256": original_sha256,
        })
        result.circuit.source_information.update({
            "format": "KiCad native schematic (via CLI XML netlist export)",
            "path": str(source),
            "filename": source.name,
            "original_schematic": str(source),
            "sha256": original_sha256,
            "original_schematic_sha256": original_sha256,
            "netlist_sha256": xml_sha256,
            "kicad_cli": command_result["executable"],
        })
        result.circuit.provenance.update({
            "kicad_cli": command_result["executable"],
            "kicad_version": version,
            "variant": variant,
            "cli_returncode": command_result["returncode"],
        })
        if kept:
            result.circuit.provenance["generated_netlist_path"] = str(path)

    def _run_export(self, source: Path, output: Path, variant: str | None) -> dict[str, Any]:
        command = [
            self.executable, "sch", "export", "netlist",
            "--format", "kicadxml", "--output", str(output),
        ]
        if variant:
            command.extend(["--variant", str(variant)])
        command.append(str(source))
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self.timeout, check=False, cwd=str(source.parent),
            )
        except subprocess.TimeoutExpired as exc:
            raise ImporterError(
                f"KiCad CLI export timed out after {self.timeout:g}s for {source}."
            ) from exc
        except OSError as exc:
            raise ImporterError(f"Could not execute KiCad CLI at {self.executable!r}: {exc}") from exc
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "No CLI diagnostics returned.").strip()
            raise ImporterError(f"KiCad CLI export failed (exit {result.returncode}): {details}")
        if not output.is_file() or output.stat().st_size == 0:
            details = (result.stderr or result.stdout or "").strip()
            raise ImporterError(
                f"KiCad CLI returned success but did not create a readable non-empty netlist at {output}. {details}"
            )
        try:
            with output.open("rb") as stream:
                if not stream.read(1):
                    raise OSError("empty file")
        except OSError as exc:
            raise ImporterError(f"KiCad CLI output is not readable: {output}: {exc}") from exc
        return {
            "executable": self.executable,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
