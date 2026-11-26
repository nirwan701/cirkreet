"""KiCad XML netlist importer; connectivity comes only from explicit export mappings."""
from __future__ import annotations
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..circuit_ir import CircuitIR
from .base import ImportResult, IMPORTER_VERSION, ImporterError

_GROUND_NAMES = {"0", "gnd", "/gnd"}

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def _children(element, name):
    return [child for child in list(element) if _local(child.tag) == name]

def _first(element, name):
    return next(iter(_children(element, name)), None)

def _text(element, name, default=""):
    child = _first(element, name)
    return (child.text or "").strip() if child is not None and child.text else default

def _properties(comp):
    result = {}
    for node in comp.iter():
        tag = _local(node.tag)
        if tag == "property":
            name, value = node.get("name"), node.get("value")
            if name:
                result[name] = "" if value is None else value
        elif tag == "field":
            name = node.get("name")
            if name:
                result[name] = (node.text or "").strip()
    return result

def _sheet_refs(comp):
    sheetpath = _first(comp, "sheetpath")
    if sheetpath is None:
        return []
    records = []
    if sheetpath.get("names") is not None or sheetpath.get("tstamps") is not None:
        records.append({"names": sheetpath.get("names", ""), "tstamps": sheetpath.get("tstamps", "")})
    records.extend({"names": child.get("names", ""), "tstamps": child.get("tstamps", "")} for child in _children(sheetpath, "sheet"))
    return records

def _simulation_parameters(reference: str, value: str, normalized_value: float | None, properties: dict[str, str]) -> dict[str, Any]:
    """Translate basic KiCad-exported values into the solver's explicit parameter schema.

    This intentionally covers only standard two-terminal R/C/L parts and
    independent V/I sources. Other symbol families remain extraction-only until
    an explicit numerical model adapter is selected.
    """
    import re
    kind = reference[:1].upper()
    if kind in {"R", "C", "L"}:
        return {"value": float(normalized_value)} if normalized_value is not None else {}
    if kind not in {"V", "I"}:
        return {}

    # KiCad versions and symbol libraries differ in the property used to hold
    # simulator parameters. Prefer Sim.Params when present, otherwise use Value.
    source_text = next((str(v).strip() for k, v in properties.items()
                        if str(k).casefold() in {"sim.params", "simulation parameters", "simulator parameters"} and str(v).strip()), str(value).strip())
    params: dict[str, Any] = {"dc": 0.0}
    waveform = re.fullmatch(r"([A-Za-z]+)\s*\((.*)\)", source_text)
    if waveform:
        name = waveform.group(1).upper()
        args = [part.strip() for part in waveform.group(2).replace(",", " ").split() if part.strip()]
        if name in {"SIN", "PULSE", "PWL"}:
            params["waveform"] = {"kind": name, "arguments": args}
            try:
                from ..parser.units import parse_spice_number
                params["dc"] = float(parse_spice_number(args[0])) if args else 0.0
            except (TypeError, ValueError, OverflowError):
                params["dc"] = 0.0
            return params

    # Parse the common SPICE source form: [DC value] [AC magnitude [phase]].
    # This also accepts a plain numeric KiCad Value field as a DC source.
    try:
        from ..parser.units import parse_spice_number
        dc_match = re.search(r"\bDC\s+([^\s]+)", source_text, flags=re.IGNORECASE)
        ac_match = re.search(r"\bAC\s+([^\s]+)(?:\s+([^\s]+))?", source_text, flags=re.IGNORECASE)
        if dc_match:
            params["dc"] = float(parse_spice_number(dc_match.group(1)))
        elif not ac_match:
            params["dc"] = float(parse_spice_number(source_text))
        elif normalized_value is not None:
            params["dc"] = float(normalized_value)
        if ac_match:
            params["ac_magnitude"] = float(parse_spice_number(ac_match.group(1)))
            params["ac_phase_deg"] = float(parse_spice_number(ac_match.group(2))) if ac_match.group(2) else 0.0
            params["ac_specified"] = True
    except (TypeError, ValueError, OverflowError):
        if normalized_value is not None:
            params["dc"] = float(normalized_value)
        else:
            params["simulation_note"] = "Source value was not recognized; set its DC/AC or waveform parameters explicitly before simulation."
    return params


class KiCadXMLImporter:
    """Read .kicad_net XML and retain explicit component-pin-net relationships."""
    def import_file(self, path: str | Path, *, original_schematic: str | Path | None = None,
                    variant: str | None = None, kicad_version: str | None = None,
                    ai_provenance: dict[str, Any] | None = None) -> ImportResult:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise ImporterError(f"KiCad XML netlist does not exist: {p}")
        try:
            raw = p.read_bytes()
            root = ET.fromstring(raw)
        except (OSError, ET.ParseError) as exc:
            raise ImporterError(f"Cannot read KiCad XML netlist {p}: {exc}") from exc
        if _local(root.tag) != "export":
            raise ImporterError("The XML root is not a KiCad <export> element; refusing to interpret arbitrary XML as a KiCad netlist.")
        digest = hashlib.sha256(raw).hexdigest()
        original = Path(original_schematic).expanduser().resolve() if original_schematic else None
        ir = self._build_ir(root, p, digest, original, variant, kicad_version, ai_provenance)
        metadata = {
            "format": "kicad_xml", "filename": p.name, "path": str(p), "sha256": digest,
            "original_schematic": str(original) if original else None, "variant": variant,
            "kicad_version": kicad_version, "simulation_executed": False,
        }
        diagnostics = list(ir.validation_results.get("errors", [])) + list(ir.validation_results.get("warnings", []))
        return ImportResult(ir, metadata, diagnostics)

    def _build_ir(self, root, path, digest, original, variant, kicad_version, ai_provenance):
        ir = CircuitIR()
        design = _first(root, "design")
        tool = _text(design, "tool") if design is not None else ""
        source = _text(design, "source") if design is not None else ""
        ir.source_information = {
            "format": "KiCad XML netlist", "path": str(path), "filename": path.name,
            "sha256": digest, "source": source, "design_date": _text(design, "date") if design is not None else None,
            "original_schematic": str(original) if original else None,
            "variant": variant, "kicad_version": kicad_version,
        }
        # Preserve the sheets KiCad actually exported. Per-component sheet paths
        # are retained separately in each component's source_references field.
        sheets = _first(design, "sheets") if design is not None else None
        if sheets is not None:
            ir.hierarchy = [
                {"kind": "sheet", "number": sheet.get("number"),
                 "name": sheet.get("name"), "tstamps": _text(sheet, "tstamps") or None}
                for sheet in _children(sheets, "sheet")
            ]
        ports = _first(root, "ports")
        if ports is not None:
            ir.ports = [
                {"attributes": dict(port.attrib),
                 "children": [{"tag": _local(child.tag), "attributes": dict(child.attrib),
                               "text": (child.text or "").strip() or None}
                              for child in list(port)]}
                for port in list(ports)
            ]
        ir.provenance = {
            "parser": "CirKreety KiCadXMLImporter", "importer_version": IMPORTER_VERSION,
            "connectivity_basis": "explicit KiCad XML net/node pin mapping; no geometric inference",
            "kicad_export_tool": tool or None,
        }
        if ai_provenance:
            ir.provenance["external_generation"] = dict(ai_provenance)

        # Library symbol pin inventories are the only basis for asserting that an
        # absent pin is unconnected. Without this information, the pin is unknown.
        libpin_map = {}
        libparts = _first(root, "libparts")
        if libparts is not None:
            for libpart in _children(libparts, "libpart"):
                key = (libpart.get("lib", ""), libpart.get("part", ""))
                pins_node = _first(libpart, "pins")
                pins = []
                if pins_node is not None:
                    for pin in _children(pins_node, "pin"):
                        number = pin.get("num") or pin.get("number")
                        if number is not None:
                            pins.append({"number": str(number), "name": pin.get("name") or str(number),
                                         "type": pin.get("type"), "source": "KiCad libpart pin inventory"})
                libpin_map[key] = pins

        components_parent = _first(root, "components")
        comp_elements = _children(components_parent, "comp") if components_parent is not None else []
        components_by_ref = {}
        duplicate_refs = set()
        component_source = {}
        for index, comp_element in enumerate(comp_elements, 1):
            ref = (comp_element.get("ref") or "").strip()
            if not ref:
                ref = f"__MISSING_REF_{index}"
                ir.validation_results.setdefault("errors", []).append({"code": "MISSING_REFERENCE", "component": ref, "message": "Component reference is missing in the source export."})
            if ref.casefold() in components_by_ref:
                duplicate_refs.add(ref)
                ir.validation_results.setdefault("errors", []).append({"code": "DUPLICATE_REFERENCE", "component": ref, "message": "Multiple component records use the same reference designator."})
            libsource = _first(comp_element, "libsource")
            lib_id = ""
            if libsource is not None:
                lib = libsource.get("lib", "")
                part = libsource.get("part", "")
                lib_id = f"{lib}:{part}" if lib and part else (part or lib)
            else:
                lib, part = "", ""
            properties = _properties(comp_element)
            value = _text(comp_element, "value", properties.get("Value", ""))
            try:
                from ..parser.units import parse_spice_number
                normalized_value = parse_spice_number(value) if value else None
            except (TypeError, ValueError, OverflowError):
                normalized_value = None
            footprint = _text(comp_element, "footprint", properties.get("Footprint", ""))
            datasheet = _text(comp_element, "datasheet", properties.get("Datasheet", ""))
            pins = libpin_map.get((lib, part), [])
            component = {
                "id": f"cmp_{index:04d}", "reference": ref,
                "type": ref[:1].upper() if ref else "?", "symbol_type": lib_id or (ref[:1].upper() if ref else "?"),
                "category": _category(ref[:1].upper() if ref else "?"),
                "value": value, "value_raw": value, "normalized_value": normalized_value,
                "parameters": _simulation_parameters(ref, value, normalized_value, properties),
                "lib_id": lib_id, "properties": properties,
                "footprint": footprint, "datasheet": datasheet,
                "pins": [], "pin_inventory_complete": bool(pins),
                "source_references": {"sheet_path": _sheet_refs(comp_element),
                                      "tstamps": _text(comp_element, "tstamps") or None,
                                      "source": source or None},
                "validation_status": "extracted_explicitly",
            }
            if not value:
                ir.validation_results.setdefault("warnings", []).append({"code": "MISSING_COMPONENT_VALUE", "component": ref, "message": "Component has no exported value."})
            if not lib_id or not pins:
                ir.validation_results.setdefault("warnings", []).append({"code": "UNKNOWN_OR_INCOMPLETE_LIBRARY_SYMBOL", "component": ref, "lib_id": lib_id or None, "message": "No complete library pin inventory was present; unlisted pins are unknown, not assumed unconnected."})
            components_by_ref.setdefault(ref.casefold(), component)
            component_source[ref.casefold()] = comp_element
            ir.components.append(component)

        connected: dict[tuple[str, str], dict] = {}
        # KiCad XML netlists encode explicit no-connect markers as a net node
        # whose pintype contains the +no_connect suffix. Older/custom exports
        # may also provide a dedicated no_connects section; support both.
        no_connect: set[tuple[str, str]] = set()
        no_connect_evidence: dict[tuple[str, str], dict] = {}
        net_records = []
        net_names_seen = {}
        nets_parent = _first(root, "nets")
        for net_index, net in enumerate(_children(nets_parent, "net") if nets_parent is not None else [], 1):
            name = (net.get("name") or f"N${net_index}").strip()
            code = net.get("code")
            net_key = f"{code or net_index}:{name}"
            net_nodes = _children(net, "node")
            conductive_nodes = []
            for node in net_nodes:
                ref = (node.get("ref") or "").strip()
                pin_no = str(node.get("pin") or node.get("pin_number") or "")
                pin_type = (node.get("pintype") or "").casefold()
                if "+no_connect" in pin_type or pin_type == "no_connect":
                    if ref and pin_no:
                        key = (ref.casefold(), pin_no)
                        no_connect.add(key)
                        no_connect_evidence[key] = {
                            "kind": "KiCad XML no-connect pin marker",
                            "net_name": name, "net_code": code,
                            "ref": ref, "pin": pin_no,
                            "pintype": node.get("pintype"),
                        }
                    continue
                conductive_nodes.append(node)

            # A net made exclusively from no-connect marker nodes is not an
            # electrical net. Do not create a CircuitIR node for it.
            if not conductive_nodes:
                continue
            if name.casefold() in net_names_seen:
                ir.validation_results.setdefault("warnings", []).append({"code": "DUPLICATE_NET_NAME", "net": name, "message": "The export contains multiple net records with the same display name; stable IDs remain distinct."})
            net_names_seen[name.casefold()] = net_key
            is_ground = name.casefold() in _GROUND_NAMES
            node_id = 0 if is_ground else len([n for n in ir.nodes if n["id"] != 0]) + 1
            if is_ground:
                ground = next((n for n in ir.nodes if n["id"] == 0), None)
                if ground is None:
                    ir.nodes.append({"id": 0, "name": "0", "aliases": [name] if name != "0" else [], "source_name": name, "is_reference": True, "source_present": True})
                elif name not in ground["aliases"] and name != "0":
                    ground["aliases"].append(name)
            else:
                ir.nodes.append({"id": node_id, "name": name, "aliases": [], "source_code": code, "is_reference": False, "source_present": True})
            ir.labels.append({"name": name, "node_id": node_id, "net_code": code, "net_key": net_key})
            net_connections = []
            for node in conductive_nodes:
                ref = (node.get("ref") or "").strip()
                pin_no = str(node.get("pin") or node.get("pin_number") or "")
                if ref.casefold() not in components_by_ref:
                    ir.validation_results.setdefault("errors", []).append({"code": "CONNECTIVITY_UNKNOWN_COMPONENT", "component": ref or None, "pin": pin_no or None, "net": name, "message": "A net entry references a component that is absent from the component inventory."})
                    continue
                key = (ref.casefold(), pin_no)
                evidence = {"kind": "KiCad net node", "net": name, "net_code": code, "ref": ref, "pin": pin_no}
                if key in connected and connected[key]["node_id"] != node_id:
                    ir.validation_results.setdefault("errors", []).append({"code": "PIN_ASSIGNED_TO_MULTIPLE_NETS", "component": ref, "pin": pin_no, "nets": [connected[key]["node_name"], name], "message": "One component pin is assigned to more than one net."})
                else:
                    connected[key] = {"node_id": node_id, "node_name": name,
                                      "name": node.get("pinfunction") or pin_no,
                                      "pintype": node.get("pintype"), "source_evidence": evidence}
                net_connections.append({"reference": ref, "pin": pin_no,
                                        "pin_name": node.get("pinfunction") or pin_no,
                                        "source_evidence": evidence})
            net_records.append({"node_id": node_id, "name": name, "net_code": code,
                                "is_reference": is_ground, "connections": net_connections,
                                "hierarchy": _net_hierarchy(name)})

        nc_parent = _first(root, "no_connects")
        if nc_parent is not None:
            for element in nc_parent.iter():
                tag = _local(element.tag)
                if tag in {"node", "no_connect"}:
                    ref, pin = element.get("ref"), element.get("pin")
                    if ref and pin is not None:
                        key = (ref.casefold(), str(pin))
                        no_connect.add(key)
                        no_connect_evidence.setdefault(key, {
                            "kind": "KiCad explicit no-connect marker", "ref": ref, "pin": str(pin)
                        })

        for component in ir.components:
            refkey = component["reference"].casefold()
            comp_element = component_source.get(refkey)
            libsource = _first(comp_element, "libsource") if comp_element is not None else None
            lib, part = (libsource.get("lib", ""), libsource.get("part", "")) if libsource is not None else ("", "")
            inventory = libpin_map.get((lib, part), [])
            inventory_by_no = {p["number"]: p for p in inventory}
            encountered = set()
            all_pin_numbers = set(inventory_by_no)
            all_pin_numbers.update(pin for r, pin in connected if r == refkey)
            all_pin_numbers.update(pin for r, pin in no_connect if r == refkey)
            for pin_no in sorted(all_pin_numbers, key=_pin_sort):
                conn = connected.get((refkey, pin_no))
                marked_nc = (refkey, pin_no) in no_connect
                inventory_pin = inventory_by_no.get(pin_no)
                if conn and marked_nc:
                    ir.validation_results.setdefault("errors", []).append({"code": "PIN_CONNECTED_AND_NO_CONNECT", "component": component["reference"], "pin": pin_no, "message": "The same pin is mapped to a net and marked no-connect."})
                if conn:
                    status = "connected"
                elif marked_nc:
                    status = "no_connect"
                elif inventory_pin:
                    status = "unconnected"
                    ir.validation_results.setdefault("warnings", []).append({"code": "UNCONNECTED_PIN", "component": component["reference"], "pin": pin_no, "message": "Complete library inventory proves this pin has no net mapping and no explicit no-connect marker."})
                else:
                    status = "unknown"
                if conn and inventory and pin_no not in inventory_by_no:
                    ir.validation_results.setdefault("errors", []).append({"code": "PIN_NOT_IN_LIBRARY_INVENTORY", "component": component["reference"], "pin": pin_no, "message": "Mapped pin is absent from the exported library pin inventory."})
                pin_data = inventory_pin or {}
                pin = {
                    "number": pin_no, "name": (pin_data.get("name") or (conn or {}).get("name") or pin_no),
                    "type": pin_data.get("type") or (conn or {}).get("pintype"),
                    "node_id": conn["node_id"] if conn else None,
                    "node_name": conn["node_name"] if conn else None,
                    "connectivity_status": status,
                    "source_evidence": ((conn or {}).get("source_evidence") if conn else
                                        no_connect_evidence.get((refkey, pin_no), {"kind": "KiCad no-connect marker", "ref": component["reference"], "pin": pin_no}) if marked_nc else
                                        {"kind": "KiCad library pin inventory" if inventory_pin else "insufficient source pin inventory", "ref": component["reference"], "pin": pin_no}),
                }
                component["pins"].append(pin)
                ir.terminals.append({"component": component["reference"], **pin})
                encountered.add(pin_no)
            component["pin_inventory_complete"] = bool(inventory)
            component["extraction_status"] = "complete_pin_inventory" if inventory else "partial_pin_inventory"

        # Build CircuitIR node records with explicit net membership retained.
        if not any(n["id"] == 0 for n in ir.nodes):
            ir.nodes.insert(0, {"id": 0, "name": "0", "aliases": [], "is_reference": True, "source_present": False})
        ir.nodes.sort(key=lambda n: n["id"])
        ir.source_information.update({"component_count": len(ir.components), "net_count": len(net_records), "net_records": net_records})
        for i, component in enumerate(ir.components, 1):
            node_ids = [p["node_id"] for p in component["pins"] if p.get("node_id") is not None]
            if len(node_ids) >= 2:
                ir.branches.append({"id": f"br_{i:04d}", "component": component["reference"], "terminal_nodes": node_ids, "branch_unknown": "solver-dependent"})
        errors = ir.validation_results.get("errors", [])
        warnings = ir.validation_results.get("warnings", [])
        ir.validation_results.update({
            "status": "error" if errors else "warning" if warnings else "success",
            "errors": errors, "warnings": warnings,
            "connectivity": {"basis": "explicit_export_mapping", "net_count": len(net_records),
                             "connected_pin_count": sum(1 for t in ir.terminals if t.get("connectivity_status") == "connected"),
                             "no_connect_pin_count": sum(1 for t in ir.terminals if t.get("connectivity_status") == "no_connect"),
                             "unconnected_pin_count": sum(1 for t in ir.terminals if t.get("connectivity_status") == "unconnected"),
                             "unknown_pin_count": sum(1 for t in ir.terminals if t.get("connectivity_status") == "unknown")},
        })
        return ir

def _pin_sort(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value.casefold())

def _category(kind: str) -> str:
    if kind in "RCL": return "passive"
    if kind in "DQJM": return "semiconductor"
    if kind in "VI": return "source"
    if kind in "UE": return "analog"
    return "other"

def _net_hierarchy(name: str):
    return [part for part in name.split("/")[:-1] if part] if name.startswith("/") else []
