"""Documented, conservative SPICE netlist subset parser."""
from __future__ import annotations
import re
import shlex
from pathlib import Path
from ..circuit_ir import CircuitIR
from .units import parse_spice_number, UNIT_FOR_KIND

GROUND_NAMES = {"0", "gnd", "gnd!", "ground"}


def _logical_lines(text: str):
    acc = ""
    start = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith(("*", "//")):
            continue
        # Inline $ comment; semicolon is a SPICE comment delimiter outside parens.
        line = line.split("$")[0].rstrip()
        if not line.strip(): continue
        if line.lstrip().startswith("+"):
            acc += " " + line.lstrip()[1:].strip()
        else:
            if acc: yield start, acc
            acc, start = line.strip(), lineno
    if acc: yield start, acc


def _pairs(text: str, params: dict):
    out = {}
    for key, val in re.findall(r"([A-Za-z_][\w]*)\s*=\s*(\{[^}]+\}|[^\s,)]+)", text):
        try: out[key.upper()] = parse_spice_number(val, params)
        except Exception: out[key.upper()] = val
    return out


def _split_tokens(line: str):
    # Keep parenthesized waveform blocks and brace expressions intact.
    return re.findall(r"\([^)]*\)|\{[^}]*\}|[^\s]+", line)



def _substitute_braced_expressions(line: str, parameters: dict[str, float]) -> str:
    def replace(match):
        expression = match.group(1)
        try:
            return format(parse_spice_number(expression, parameters), ".17g")
        except Exception as exc:
            raise ValueError(f"Unsupported subcircuit parameter expression {{{expression}}}: {exc}") from exc
    return re.sub(r"\{([^{}]+)\}", replace, line)


def _parse_assignments(tokens: list[str], parameters: dict[str, float]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for token in tokens:
        if token.casefold() in ("params:", "params"):
            continue
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        try:
            resolved[key.upper()] = parse_spice_number(raw_value, parameters)
        except Exception as exc:
            raise ValueError(f"Unsupported subcircuit parameter {key}={raw_value}: {exc}") from exc
    return resolved


def _expand_subcircuits(text: str):
    """Flatten positional X instances with local nodes, models, and parameters.

    This is deliberately a conservative structural expansion, not a complete
    SPICE macro language.  It supports nested positional subcircuits, simple
    numeric .param defaults/overrides, and local .model name scoping.
    """
    logical = list(_logical_lines(text))
    definitions: dict[str, dict] = {}
    top_level: list[tuple[int, str]] = []
    current = None
    for lineno, line in logical:
        lower = line.lower()
        if lower.startswith(".subckt"):
            if current is not None:
                raise ValueError(f"Nested .SUBCKT declaration at line {lineno} before .ENDS for {current}.")
            tokens = _split_tokens(line)
            if len(tokens) < 2:
                raise ValueError(f"Invalid .SUBCKT declaration at line {lineno}.")
            name = tokens[1].upper()
            if name in definitions:
                raise ValueError(f"Duplicate .SUBCKT definition {name!r} at line {lineno}.")
            header = tokens[2:]
            pin_tokens = []
            default_tokens = []
            params_started = False
            for token in header:
                if token.casefold() in ("params:", "params"):
                    params_started = True
                    continue
                if "=" in token:
                    params_started = True
                (default_tokens if params_started else pin_tokens).append(token)
            current = name
            definitions[name] = {"name": name, "pins": pin_tokens, "defaults": default_tokens,
                                 "body": [], "source_line": lineno, "raw_header": line}
        elif lower.startswith(".ends"):
            if current is None:
                raise ValueError(f".ENDS at line {lineno} has no matching .SUBCKT.")
            current = None
        elif current is not None:
            definitions[current]["body"].append({"line": lineno, "raw": line})
        else:
            top_level.append((lineno, line))
    if current is not None:
        raise ValueError(f".SUBCKT {current} is missing .ENDS.")

    # A netlist that only declares reusable definitions keeps its original
    # source representation; the normal parser records those declarations.
    known_instances = [
        (lineno, line) for lineno, line in top_level
        if line.split(None, 1)[0][:1].upper() == "X"
        and any(token.upper() in definitions for token in _split_tokens(line)[1:])
    ]
    if not known_instances:
        return text, {}, {}, definitions

    global_parameters: dict[str, float] = {}
    for _, line in top_level:
        if line.lower().startswith(".param"):
            for key, value in _pairs(line[6:], global_parameters).items():
                if isinstance(value, (int, float)):
                    global_parameters[key] = float(value)

    node_counts = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "Q": 3,
                   "J": 3, "M": 4, "E": 4, "G": 4, "F": 2, "H": 2,
                   "S": 4, "W": 3, "B": 2, "T": 4, "K": 0}
    generated_hierarchy: dict[str, list[str]] = {}
    hierarchy_records: list[dict] = []

    def expand_instance(line: str, parent_node_map: dict[str, str], prefix: str,
                        hierarchy: list[str], stack: list[str], depth: int):
        if depth > 16:
            raise ValueError("Subcircuit expansion exceeded maximum depth 16.")
        tokens = _split_tokens(line)
        instance_ref = tokens[0]
        matches = [index for index, token in enumerate(tokens[1:], 1) if token.upper() in definitions]
        if not matches:
            return [line]
        subckt_index = matches[-1]
        definition_name = tokens[subckt_index].upper()
        if definition_name in stack:
            chain = " -> ".join(stack + [definition_name])
            raise ValueError(f"Recursive subcircuit instantiation detected: {chain}.")
        definition = definitions[definition_name]
        actual_nodes_raw = tokens[1:subckt_index]
        if len(actual_nodes_raw) != len(definition["pins"]):
            raise ValueError(
                f"{instance_ref}: subcircuit {definition_name} expects {len(definition['pins'])} pins, "
                f"but {len(actual_nodes_raw)} were provided."
            )
        actual_nodes = []
        for raw_node in actual_nodes_raw:
            key = raw_node.casefold()
            actual_nodes.append(parent_node_map.get(key, raw_node))
        local_node_map = {pin.casefold(): actual for pin, actual in zip(definition["pins"], actual_nodes)}
        instance_prefix = f"{prefix}_{instance_ref}" if prefix else instance_ref
        instance_hierarchy = hierarchy + [instance_ref]
        parameters = dict(global_parameters)
        parameters.update(_parse_assignments(definition["defaults"], parameters))
        parameters.update(_parse_assignments(tokens[subckt_index + 1:], parameters))

        # Resolve all local .param definitions before expanding devices so body
        # order does not change parameter meaning.
        for body_record in definition["body"]:
            raw = body_record["raw"]
            if raw.lower().startswith(".param"):
                assignments = _parse_assignments(_split_tokens(raw[6:]), parameters)
                parameters.update(assignments)

        local_model_names = {}
        local_references = {}
        for body_record in definition["body"]:
            body_tokens = _split_tokens(body_record["raw"])
            if not body_tokens:
                continue
            if body_tokens[0].lower() == ".model" and len(body_tokens) > 1:
                local_model_names[body_tokens[1].casefold()] = f"{instance_prefix}_{body_tokens[1]}"
            elif not body_tokens[0].startswith("."):
                original_ref = body_tokens[0]
                local_references[original_ref.casefold()] = (
                    f"{original_ref[0]}_{instance_prefix}_{original_ref[1:]}"
                )

        expanded_lines: list[str] = []
        expanded_components: list[str] = []
        for body_record in definition["body"]:
            raw = body_record["raw"]
            if raw.lower().startswith(".param"):
                continue
            raw = _substitute_braced_expressions(raw, parameters)
            body_tokens = _split_tokens(raw)
            if not body_tokens:
                continue
            directive = body_tokens[0].lower()
            if directive == ".model":
                if len(body_tokens) > 1:
                    body_tokens[1] = local_model_names.get(body_tokens[1].casefold(), body_tokens[1])
                expanded_lines.append(" ".join(body_tokens))
                continue
            if directive.startswith("."):
                # Preserve simple .IC node references in the expanded namespace.
                def replace_ic(match):
                    node = match.group(1)
                    mapped = local_node_map.get(node.casefold(), f"{instance_prefix}__{node}")
                    return f"V({mapped})="
                raw = re.sub(r"V\(([^)]+)\)\s*=", replace_ic, raw, flags=re.I)
                expanded_lines.append(raw)
                continue

            element_ref = body_tokens[0]
            kind = element_ref[:1].upper()
            if kind == "X":
                sub_index_candidates = [i for i, token in enumerate(body_tokens[1:], 1)
                                        if token.upper() in definitions]
                if not sub_index_candidates:
                    expanded_lines.append(" ".join(body_tokens))
                    continue
                sub_index = sub_index_candidates[-1]
                count = sub_index - 1
                nested_actual = []
                for raw_node in body_tokens[1:sub_index]:
                    nested_actual.append(local_node_map.get(raw_node.casefold(),
                                                            "0" if raw_node.casefold() in GROUND_NAMES else f"{instance_prefix}__{raw_node}"))
                nested_line = " ".join([body_tokens[0], *nested_actual, *body_tokens[sub_index:]])
                nested = expand_instance(nested_line, {}, instance_prefix, instance_hierarchy,
                                         stack + [definition_name], depth + 1)
                expanded_lines.extend(nested)
                continue

            count = node_counts.get(kind)
            if count is None:
                expanded_lines.append(" ".join(body_tokens))
                continue
            renamed_ref = local_references.get(
                element_ref.casefold(), f"{element_ref[0]}_{instance_prefix}_{element_ref[1:]}"
            )
            body_tokens[0] = renamed_ref
            for index in range(1, min(1 + count, len(body_tokens))):
                node = body_tokens[index]
                body_tokens[index] = local_node_map.get(
                    node.casefold(), "0" if node.casefold() in GROUND_NAMES else f"{instance_prefix}__{node}"
                )
            if kind in ("F", "H") and len(body_tokens) > count + 1:
                control_ref = body_tokens[count + 1]
                body_tokens[count + 1] = local_references.get(control_ref.casefold(), control_ref)
            if kind in ("D", "Q", "J", "M", "S") and len(body_tokens) > count + 1:
                model_ref = body_tokens[count + 1]
                body_tokens[count + 1] = local_model_names.get(model_ref.casefold(), model_ref)
            expanded_lines.append(" ".join(body_tokens))
            expanded_components.append(renamed_ref)
            generated_hierarchy[renamed_ref] = instance_hierarchy

        hierarchy_records.append({
            "instance": instance_prefix, "subcircuit": definition_name,
            "pin_mapping": {pin: actual for pin, actual in zip(definition["pins"], actual_nodes)},
            "generated_components": expanded_components, "depth": depth,
        })
        return expanded_lines

    output_lines: list[str] = []
    for _, line in top_level:
        tokens = _split_tokens(line)
        if tokens and tokens[0][:1].upper() == "X" and any(token.upper() in definitions for token in tokens[1:]):
            output_lines.extend(expand_instance(line, {}, "", [], [], 1))
        else:
            output_lines.append(line)
    definition_metadata = {
        name: {"pins": definition["pins"], "defaults": definition["defaults"],
               "lines": definition["body"], "source_line": definition["source_line"]}
        for name, definition in definitions.items()
    }
    return "\n".join(output_lines), generated_hierarchy, {"instances": hierarchy_records}, definition_metadata


def parse_spice_text(text: str, source_name: str = "<memory>") -> CircuitIR:
    original_text = text
    text, expanded_hierarchy, hierarchy_metadata, subcircuit_definitions = _expand_subcircuits(text)
    ir = CircuitIR()
    ir.source_information = {"format": "SPICE-compatible netlist subset", "path": source_name, "source_text": original_text}
    params: dict[str, float] = {}
    model_defs: dict[str, dict] = {}
    raw_components = []
    active_subckt = None
    for lineno, line in _logical_lines(text):
        low = line.lower()
        if low.startswith(".title"):
            ir.metadata["title"] = line[6:].strip(); ir.directives.append({"name":".title","line":lineno,"raw":line}); continue
        if low.startswith(".param"):
            for k,v in _pairs(line[6:], params).items():
                if isinstance(v, (int,float)): params[k] = float(v)
            ir.parameters.update(params); ir.directives.append({"name":".param","line":lineno,"raw":line}); continue
        if low.startswith(".model"):
            m = re.match(r"\.model\s+(\S+)\s+(\S+)\s*(?:\((.*)\))?", line, re.I)
            if m:
                name, typ, ptxt = m.groups()
                model_defs[name.upper()] = {"type":typ.upper(), "parameters":_pairs(ptxt or "", params), "source_line":lineno, "raw":line}
                ir.models[name] = model_defs[name.upper()]
            else:
                ir.validation_results.setdefault("warnings", []).append({"code":"MODEL_PARSE","line":lineno,"message":"Could not parse .model statement."})
            ir.directives.append({"name":".model","line":lineno,"raw":line}); continue
        if low.startswith(".subckt"):
            m = re.match(r"\.subckt\s+(\S+)\s*(.*)", line, re.I)
            if m:
                active_subckt=m.group(1).upper(); ir.subcircuits[active_subckt]={"pins":m.group(2).split(),"lines":[],"source_line":lineno}
            ir.directives.append({"name":".subckt","line":lineno,"raw":line}); continue
        if low.startswith(".ends"):
            active_subckt=None; ir.directives.append({"name":".ends","line":lineno,"raw":line}); continue
        if active_subckt:
            ir.subcircuits[active_subckt]["lines"].append({"line":lineno,"raw":line}); continue
        if low.startswith(".include") or low.startswith(".lib"):
            ir.directives.append({"name":line.split()[0].lower(),"line":lineno,"raw":line})
            ir.validation_results.setdefault("warnings", []).append({"code":"EXTERNAL_LIBRARY_NOT_RESOLVED","line":lineno,"message":"External include/library files are recorded but are not automatically loaded."})
            continue
        if low.startswith(".end"):
            ir.directives.append({"name":".end","line":lineno,"raw":line}); continue
        if line.startswith("."):
            directive = line.split()[0].lower()
            ir.directives.append({"name":directive,"line":lineno,"raw":line})
            tokens = _split_tokens(line)
            try:
                if directive == ".op": ir.analysis_configuration["dc_operating_point"]={"requested":True}
                elif directive == ".ac" and len(tokens)>=5:
                    # .ac dec|oct|lin points start stop
                    ir.analysis_configuration["ac_sweep"]={"scale":tokens[1].lower(),"points":int(tokens[2]),"start_Hz":parse_spice_number(tokens[3],params),"stop_Hz":parse_spice_number(tokens[4],params)}
                elif directive == ".dc" and len(tokens)>=5:
                    ir.analysis_configuration["dc_sweep"]={"source":tokens[1],"start":parse_spice_number(tokens[2],params),"stop":parse_spice_number(tokens[3],params),"step":parse_spice_number(tokens[4],params)}
                elif directive == ".tran" and len(tokens)>=3:
                    ir.analysis_configuration["transient"]={"step_s":parse_spice_number(tokens[1],params),"stop_s":parse_spice_number(tokens[2],params)}
                    if len(tokens) > 3:
                        ir.analysis_configuration["transient"]["unsupported_options"] = tokens[3:]
                        ir.validation_results.setdefault("warnings", []).append({
                            "code":"UNSUPPORTED_TRAN_OPTION", "line":lineno,
                            "options":tokens[3:],
                            "message":"Additional .tran options are preserved but not implemented; only TSTEP and TSTOP are supported.",
                        })
                elif directive == ".ic":
                    for node,val in re.findall(r"V\(([^)]+)\)\s*=\s*([^\s]+)",line,re.I): ir.initial_conditions[node]=parse_spice_number(val,params)
                elif directive == ".options": ir.simulation_options.update(_pairs(line[len(tokens[0]):],params))
                elif directive == ".circhecker":
                    custom=_pairs(line[len(tokens[0]):],params)
                    for key,value in custom.items():
                        if key == "AC_INPUT": ir.analysis_configuration["ac_input_node"]=str(value)
                        elif key == "AC_OUTPUT": ir.analysis_configuration["ac_output_node"]=str(value)
                        elif key == "OUTPUT": ir.analysis_configuration["output_node"]=str(value)
                        elif key == "OUTPUT_NEG": ir.analysis_configuration["output_negative_node"]=str(value)
                        else: ir.validation_results.setdefault("warnings", []).append({"code":"UNKNOWN_CIRCHECKER_OPTION","line":lineno,"option":key,"message":"Unknown .circhecker analysis option was preserved but not applied."})
                elif directive == ".nodeset": ir.directives[-1]["parsed_status"]="recorded_not_enforced"
                elif directive in (".save",".probe",".meas",".four",".tf"): ir.directives[-1]["parsed_status"]="recorded_not_executed"
                else: ir.validation_results.setdefault("warnings", []).append({"code":"UNSUPPORTED_DIRECTIVE","line":lineno,"message":f"Directive {directive} is preserved but not executed."})
            except Exception as exc:
                ir.validation_results.setdefault("warnings", []).append({"code":"DIRECTIVE_PARSE","line":lineno,"message":str(exc),"raw":line})
            continue
        tokens = _split_tokens(line)
        if not tokens: continue
        ref = tokens[0]; kind=ref[0].upper()
        record={"reference":ref,"type":kind,"raw":line,"source_line":lineno,"tokens":tokens[1:]}
        raw_components.append(record)
    # resolve nodes and element shapes
    node_names=[]
    for item in raw_components:
        t=item["tokens"]; k=item["type"]
        ncount={"R":2,"C":2,"L":2,"V":2,"I":2,"D":2,"Q":3,"J":3,"M":4,"E":4,"G":4,"F":2,"H":2,"S":4,"W":3,"K":0,"B":2,"T":4}.get(k)
        if ncount is None:
            item["unsupported_parse"] = f"Unknown element prefix {k}"
            continue
        if len(t)<ncount:
            item["unsupported_parse"] = f"Expected at least {ncount} terminal/control tokens."
            continue
        item["nodes"] = t[:ncount]
        for n in item["nodes"]:
            if n.lower() not in GROUND_NAMES and n not in node_names: node_names.append(n)
    node_id={"0":0}
    for n in node_names: node_id[n]=len(node_id)
    ir.nodes=[{"id":0,"name":"0","aliases":["GND"]}]+[{"id":node_id[n],"name":n,"aliases":[]} for n in node_names]
    for item in raw_components:
        ref,k,t=item["reference"],item["type"],item["tokens"]
        if item.get("unsupported_parse"):
            ir.validation_results.setdefault("errors",[]).append({"code":"ELEMENT_PARSE","line":item["source_line"],"component":ref,"message":item["unsupported_parse"]}); continue
        nodes=[0 if n.lower() in GROUND_NAMES else node_id[n] for n in item.get("nodes",[])]
        model_ref=None; raw_value=None; params_comp={}; waveform=None; unit=UNIT_FOR_KIND.get(k)
        try:
            if k in "RCLVI":
                rest=t[2:]
                if k in "VI":
                    tail=" ".join(rest)
                    mv=re.search(r"\bDC\s+([^\s]+)",tail,re.I)
                    if mv: raw_value=mv.group(1)
                    elif rest and not rest[0].startswith("AC") and not rest[0].startswith(("SIN","PULSE","PWL","EXP")): raw_value=rest[0]
                    else: raw_value="0"
                    dc=parse_spice_number(raw_value,params)
                    ma=re.search(r"\bAC\s+([^\s]+)(?:\s+([^\s]+))?",tail,re.I)
                    ac_mag=parse_spice_number(ma.group(1),params) if ma else (1.0 if k=="V" else 0.0)
                    ac_phase=parse_spice_number(ma.group(2),params) if ma and ma.group(2) else 0.0
                    wm=re.search(r"\b(SIN|PULSE|PWL|EXP)\s*(\([^)]*\))",tail,re.I)
                    if wm: waveform={"kind":wm.group(1).upper(),"arguments":wm.group(2)[1:-1].split()}
                    params_comp={"dc":dc,"ac_magnitude":ac_mag,"ac_phase_deg":ac_phase,"ac_specified":bool(ma),"waveform":waveform}
                else:
                    raw_value=rest[0] if rest else None
                    val=parse_spice_number(raw_value,params)
                    params_comp={"value":val,"raw_value":raw_value,"unit":unit}
                    params_comp.update(_pairs(" ".join(rest[1:]),params))
            elif k in "DQJM":
                model_ref=t[len(nodes)] if len(t)>len(nodes) else None
                model=model_defs.get(str(model_ref).upper(),{})
                params_comp={"model_type":model.get("type"),"model_parameters":model.get("parameters",{}),"model_ref":model_ref}
                params_comp.update(_pairs(" ".join(t[len(nodes)+1:]),params))
                if k=="D": params_comp.setdefault("IS",model.get("parameters",{}).get("IS",1e-14)); params_comp.setdefault("N",model.get("parameters",{}).get("N",1.0))
                elif k=="Q": params_comp.setdefault("BF",model.get("parameters",{}).get("BF",100.0)); params_comp.setdefault("IS",model.get("parameters",{}).get("IS",1e-15))
                elif k=="M":
                    params_comp.setdefault("VTO",model.get("parameters",{}).get("VTO",1.0)); params_comp.setdefault("KP",model.get("parameters",{}).get("KP",1e-3)); params_comp.setdefault("LAMBDA",model.get("parameters",{}).get("LAMBDA",0.0))
                    params_comp["instance_parameters"]=_pairs(" ".join(t[len(nodes)+1:]),params)
                elif k=="J":
                    params_comp.setdefault("VTO",model.get("parameters",{}).get("VTO",-2.0)); params_comp.setdefault("BETA",model.get("parameters",{}).get("BETA",1e-3)); params_comp.setdefault("IS",model.get("parameters",{}).get("IS",1e-14))
            elif k in "EG":
                raw_value=t[len(nodes)]
                params_comp={"gain":parse_spice_number(raw_value,params),"raw_value":raw_value}
                params_comp.update(_pairs(" ".join(t[len(nodes)+1:]),params))
            elif k=="S":
                model_ref=t[len(nodes)] if len(t)>len(nodes) else None
                model=model_defs.get(str(model_ref).upper(),{})
                params_comp={"model_ref":model_ref,"model_parameters":model.get("parameters",{})}
            elif k=="K":
                params_comp={"inductor1":t[0],"inductor2":t[1],"coupling":parse_spice_number(t[2],params)}
            elif k in "FH":
                params_comp={"control_source":t[len(nodes)],"gain":parse_spice_number(t[len(nodes)+1],params)}
            elif k=="X":
                params_comp={"subcircuit":t[-1].upper(),"instance_nodes":t[:-1]}
            else:
                params_comp={"raw_parameters":" ".join(t[len(nodes):])}
            cap_names={"R":("1","2"),"C":("1","2"),"L":("1","2"),"V":("+","-"),"I":("+","-"),"D":("A","K"),"Q":("C","B","E"),"J":("D","G","S"),"M":("D","G","S","B"),"E":("+","-","CP","CN"),"G":("+","-","CP","CN"),"F":("+","-"),"H":("+","-"),"S":("+","-","CP","CN"),"K":(),"X":()}.get(k,tuple(str(i+1) for i in range(len(nodes))))
            pins=[]
            for ix,(node,pname) in enumerate(zip(nodes,cap_names)):
                pin={"number":str(ix+1),"name":pname,"node_id":node,"node_name":ir.nodes[node]["name"]}
                pins.append(pin); ir.terminals.append({"component":ref,**pin})
            component={"id":f"cmp_{len(ir.components)+1:04d}","reference":ref,"type":k,"category":{"R":"passive","C":"passive","L":"passive","V":"source","I":"source","D":"semiconductor","Q":"semiconductor","J":"semiconductor","M":"semiconductor","E":"controlled_source","G":"controlled_source","K":"magnetic","S":"switch","X":"subcircuit"}.get(k,"other"),"pins":pins,"model_ref":model_ref,"parameters":params_comp,"units":unit,"original_expression":item["raw"],"resolved_values":params_comp,"source_line":item["source_line"],"hierarchy":[],"capabilities":[],"validation_status":"parsed"}
            ir.components.append(component)
            if len(nodes)>=2 and k not in "KX": ir.branches.append({"id":f"br_{len(ir.branches)+1:04d}","component":ref,"terminal_nodes":nodes,"branch_unknown":"solver-dependent"})
        except Exception as exc:
            ir.validation_results.setdefault("errors",[]).append({"code":"VALUE_PARSE","line":item["source_line"],"component":ref,"message":str(exc),"raw":item["raw"]})
    ir.parameters.update(params)
    ir.models.update({k:v for k,v in model_defs.items()})
    # Preserve unknown/unimplemented component types as parsed-but-unsupported.
    for item in raw_components:
        if item.get("unsupported_parse"):
            ir.components.append({"id":f"cmp_{len(ir.components)+1:04d}","reference":item["reference"],"type":item["type"],"category":"unsupported","pins":[],"parameters":{},"original_expression":item["raw"],"source_line":item["source_line"],"validation_status":"unsupported"})
    ir.source_information["component_count"]=len(ir.components)
    if expanded_hierarchy:
        ir.subcircuits.update(subcircuit_definitions)
        ir.hierarchy.extend(hierarchy_metadata.get("instances", []))
        for component in ir.components:
            if component.get("reference") in expanded_hierarchy:
                component["hierarchy"] = expanded_hierarchy[component["reference"]]
        ir.provenance["subcircuit_expansion"] = {
            "status": "expanded", "instance_count": len(ir.hierarchy),
            "maximum_depth": max((int(item.get("depth", 1)) for item in ir.hierarchy), default=0),
            "method": "deterministic positional pin mapping with instance-scoped internal nodes and local model names",
        }
    ir.provenance={"parser":"circhecker.parser.spice.parse_spice_text","parser_version":"0.2","raw_netlist_preserved":True,
                   **({"subcircuit_expansion": ir.provenance.get("subcircuit_expansion")} if expanded_hierarchy else {})}
    return ir


def _select_library_section(text: str, section: str | None, source: Path) -> str:
    """Select named .LIB/.ENDL sections; without a name, include all bodies."""
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    active: str | None = None
    sectioned = False
    unsectioned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", "//")):
            if active is not None:
                sections.setdefault(active, []).append(line)
            elif not sectioned:
                unsectioned.append(line)
            continue
        tokens = _split_tokens(stripped)
        directive = tokens[0].lower() if tokens else ""
        if directive == ".lib" and len(tokens) >= 2:
            sectioned = True
            active = tokens[1].casefold()
            sections.setdefault(active, [])
            continue
        if directive == ".endl":
            active = None
            continue
        if active is not None:
            sections.setdefault(active, []).append(line)
        elif not sectioned:
            unsectioned.append(line)
    if not sectioned:
        if section is not None:
            raise ValueError(f"Requested .LIB section {section!r} is not present in {source}.")
        return text
    if section is None:
        selected = [line for records in sections.values() for line in records]
        return "\n".join(unsectioned + selected)
    key = section.casefold()
    if key not in sections:
        raise ValueError(f"Requested .LIB section {section!r} is not present in {source}.")
    return "\n".join(sections[key])


def _expand_external_files(path: Path, root: Path, stack: tuple[Path, ...],
                           records: list[dict], depth: int = 0,
                           source_text: str | None = None) -> str:
    """Expand .INCLUDE and file-form .LIB directives inside a confined tree."""
    if depth > 8:
        raise ValueError("External SPICE include depth exceeded the maximum of 8.")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"External SPICE path escapes the input directory: {path}") from exc
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Circular SPICE include detected: {chain}")
    source = source_text if source_text is not None else resolved.read_text(encoding="utf-8", errors="replace")
    output: list[str] = []
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("*", "//")):
            output.append(raw)
            continue
        try:
            tokens = shlex.split(stripped, comments=False, posix=True)
        except ValueError as exc:
            raise ValueError(f"Invalid include directive in {resolved}: {raw}: {exc}") from exc
        directive = tokens[0].casefold() if tokens else ""
        if directive not in (".include", ".lib"):
            output.append(raw)
            continue
        if directive == ".lib" and len(tokens) == 2:
            # A one-argument .LIB is a section delimiter, not a file include.
            candidate = (resolved.parent / tokens[1])
            if not candidate.is_file():
                output.append(raw)
                continue
        elif len(tokens) < 2:
            raise ValueError(f"Missing path in {directive} directive in {resolved}: {raw}")
        elif directive == ".lib":
            candidate = resolved.parent / tokens[1]
        else:
            candidate = resolved.parent / tokens[1]
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"External SPICE path escapes the input directory: {candidate}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"SPICE include/library file does not exist: {candidate}")
        section = tokens[2] if directive == ".lib" and len(tokens) >= 3 else None
        child_text = candidate.read_text(encoding="utf-8", errors="replace")
        if directive == ".lib":
            child_text = _select_library_section(child_text, section, candidate)
        records.append({"directive": directive, "path": str(candidate), "section": section})
        expanded = _expand_external_files(
            candidate, root, (*stack, resolved), records, depth + 1, source_text=child_text
        )
        output.append(expanded)
    return "\n".join(output)


def parse_spice_file(path: str | Path) -> CircuitIR:
    p = Path(path).resolve()
    original = p.read_text(encoding="utf-8", errors="replace")
    included: list[dict] = []
    expanded = _expand_external_files(p, p.parent, tuple(), included)
    ir = parse_spice_text(expanded, str(p))
    ir.source_information["source_text"] = original
    ir.source_information["included_files"] = included
    ir.provenance["external_files"] = included
    return ir
