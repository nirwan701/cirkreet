"""Structural extraction checks; these do not judge electrical design correctness."""
from __future__ import annotations


def validate_extraction(circuit):
    errors = list(circuit.validation_results.get("errors", []))
    warnings = list(circuit.validation_results.get("warnings", []))
    components = circuit.components
    refs = {}
    for comp in components:
        ref = str(comp.get("reference") or "")
        if not ref:
            errors.append({"code": "MISSING_REFERENCE", "message": "Component reference is empty."})
        elif ref.casefold() in refs:
            errors.append({"code": "DUPLICATE_REFERENCE", "component": ref,
                           "message": "Reference designator is duplicated."})
        refs[ref.casefold()] = comp
        params = comp.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        if (comp.get("value") in (None, "") and comp.get("value_raw") in (None, "")
                and not comp.get("original_expression") and not params.get("raw_value")):
            warnings.append({"code": "MISSING_COMPONENT_VALUE", "component": ref,
                             "message": "No value or raw component expression is available."})

    node_ids = [int(n["id"]) for n in circuit.nodes if "id" in n]
    node_id_set = set(node_ids)
    if len(node_ids) != len(node_id_set):
        errors.append({"code": "DUPLICATE_NODE_ID", "message": "CircuitIR contains duplicate node IDs."})

    # Node names/aliases are display labels, not globally unique identifiers.
    # Duplicate aliases across distinct IDs are suspicious but kept intact.
    alias_owners = {}
    for node in circuit.nodes:
        node_id = int(node.get("id", -1))
        labels = [node.get("name"), *node.get("aliases", [])]
        for label in (str(v).strip() for v in labels if v not in (None, "")):
            alias_owners.setdefault(label.casefold(), set()).add(node_id)
    for alias, owners in sorted(alias_owners.items()):
        if len(owners) > 1:
            warnings.append({"code": "DUPLICATE_NET_ALIAS", "alias": alias,
                             "node_ids": sorted(owners),
                             "message": "The same display name or alias is attached to multiple net IDs; IDs remain distinct."})

    pin_seen = {}
    for comp in components:
        ref = str(comp.get("reference") or "")
        pins = comp.get("pins", [])
        if comp.get("pin_inventory_complete") is True and not pins:
            warnings.append({"code": "EMPTY_COMPLETE_PIN_INVENTORY", "component": ref,
                             "message": "The component is marked as having a complete pin inventory but has no pin records."})
        for pin in pins:
            number = str(pin.get("number", ""))
            key = (ref.casefold(), number)
            if key in pin_seen:
                errors.append({"code": "DUPLICATE_PIN_RECORD", "component": ref, "pin": number,
                               "message": "The component pin occurs more than once in the extracted inventory."})
            pin_seen[key] = pin
            node_id = pin.get("node_id")
            if node_id is not None:
                try:
                    valid_node = int(node_id) in node_id_set
                except (TypeError, ValueError, OverflowError):
                    valid_node = False
                if not valid_node:
                    errors.append({"code": "INVALID_NODE_REFERENCE", "component": ref, "pin": number,
                                   "node_id": node_id, "message": "Pin refers to a node absent from CircuitIR."})
            status = pin.get("connectivity_status")
            if status not in {None, "connected", "no_connect", "unconnected", "unknown"}:
                errors.append({"code": "INVALID_CONNECTIVITY_STATUS", "component": ref, "pin": number,
                               "status": status, "message": "Unsupported connectivity status."})
            if status == "connected" and node_id is None:
                errors.append({"code": "CONNECTED_PIN_WITHOUT_NET", "component": ref, "pin": number,
                               "message": "A pin marked connected has no node ID."})
            if status in {"no_connect", "unconnected", "unknown"} and node_id is not None:
                errors.append({"code": "NONCONNECTED_PIN_WITH_NET", "component": ref, "pin": number,
                               "status": status, "message": "A non-connected pin status cannot carry a net ID."})

    for terminal in circuit.terminals:
        ref = str(terminal.get("component", ""))
        if ref.casefold() not in refs:
            errors.append({"code": "TERMINAL_UNKNOWN_COMPONENT", "component": ref,
                           "message": "Terminal refers to a component absent from CircuitIR."})
            continue
        key = (ref.casefold(), str(terminal.get("number", "")))
        pin = pin_seen.get(key)
        if pin is None:
            errors.append({"code": "TERMINAL_UNKNOWN_PIN", "component": ref,
                           "pin": terminal.get("number"),
                           "message": "Terminal record does not have a corresponding component pin record."})
        elif pin.get("node_id") != terminal.get("node_id"):
            errors.append({"code": "TERMINAL_PIN_MAPPING_MISMATCH", "component": ref,
                           "pin": terminal.get("number"),
                           "message": "Terminal and component pin records disagree on the node mapping."})

    def unique(items):
        seen, out = set(), []
        for item in items:
            key = repr(sorted(item.items(), key=lambda pair: pair[0]))
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    errors, warnings = unique(errors), unique(warnings)
    circuit.validation_results.update({
        "status": "error" if errors else "warning" if warnings else "success",
        "errors": errors,
        "warnings": warnings,
        "scope": "structural extraction only; electrical design correctness is not assessed",
    })
    return circuit.validation_results
