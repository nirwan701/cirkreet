"""Analysis adapters and execution coordinator for the unified CircuitIR."""
from __future__ import annotations
import math, re, traceback
import numpy as np
from circhecker.solver.analog import analogcircuit
from circhecker.solver.linear import circuitsolver
from circhecker.solver.transfer_function import transferfunction
from .registry import DEFAULT_COMPONENT_REGISTRY, DEFAULT_SOLVER_REGISTRY
from ..solver import (solve_dc as solve_sparse_mna_dc, supports_circuit_ir_dc,
                      solve_ac as solve_sparse_mna_ac, frequency_sweep, supports_circuit_ir_ac,
                      solve_nonlinear_dc, solve_nonlinear_ac, supports_circuit_ir_nonlinear_dc,
                      solve_transient as solve_mna_transient, supports_circuit_ir_transient,
                      supports_circuit_ir_nonlinear_transient)


def _num(x, default=None):
    try:
        y=float(x)
        return y if math.isfinite(y) else default
    except Exception: return default


def _node_voltage(nodes, key):
    return nodes.get(key, nodes.get(str(key), 0.0))


def _source_wave(component):
    p=component["parameters"]
    dc=float(p.get("dc",0.0)); w=p.get("waveform")
    if not w: return dc
    args=[]
    from ..parser.units import parse_spice_number
    for a in w.get("arguments",[]):
        try: args.append(parse_spice_number(a))
        except Exception: args.append(a)
    kind=w["kind"]
    if kind=="SIN" and len(args)>=3:
        vo,va,f=args[:3]; td=args[3] if len(args)>3 else 0.0
        theta=args[4] if len(args)>4 else 0.0; phase=args[5] if len(args)>5 else 0.0
        def fcn(t):
            tt=max(0.0,t-td)
            return vo+va*math.sin(2*math.pi*f*tt+math.radians(phase))*math.exp(-theta*tt)
        return fcn
    if kind=="PULSE" and len(args)>=2:
        v1,v2=args[:2]; td=args[2] if len(args)>2 else 0.0
        tr=args[3] if len(args)>3 else 0.0; tf=args[4] if len(args)>4 else 0.0
        pw=args[5] if len(args)>5 else 1e-3; per=args[6] if len(args)>6 else max(pw+tr+tf,2e-3)
        def fcn(t):
            q=t-td
            if q<0:return v1
            q=q%per
            if tr>0 and q<tr:return v1+(v2-v1)*q/tr
            if q<tr+pw:return v2
            if tf>0 and q<tr+pw+tf:return v2-(v2-v1)*(q-tr-pw)/tf
            return v1
        return fcn
    if kind=="PWL" and len(args)>=4:
        pts=[]
        for i in range(0,len(args)-1,2):
            try: pts.append((float(args[i]),float(args[i+1])))
            except Exception: pass
        def fcn(t):
            if not pts:return dc
            if t<=pts[0][0]:return pts[0][1]
            for (t0,v0),(t1,v1) in zip(pts,pts[1:]):
                if t<=t1:return v0+(v1-v0)*(t-t0)/(t1-t0)
            return pts[-1][1]
        return fcn
    return dc


class CircuitExecutionEngine:
    """Run supported analyses against a single parsed CircuitIR."""
    ANALYSIS_KEYS=("dc_operating_point","dc_sweep","ac_sweep","transient","frequency_analysis","transfer_function","state_space")

    def __init__(self, circuit_ir, plot_defaults=None):
        self.ir=circuit_ir
        self.plot_defaults=plot_defaults or {}
        self._node_name_to_id={n["name"]:n["id"] for n in circuit_ir.nodes}
        self._id_to_name={n["id"]:n["name"] for n in circuit_ir.nodes}
        self._warnings=[]
        self.component_registry=DEFAULT_COMPONENT_REGISTRY
        self.solver_registry=DEFAULT_SOLVER_REGISTRY

    def _analog_model(self):
        source_nodes={}; source_defs=[]
        model=analogcircuit(self.ir.metadata.get("title","Imported circuit"))
        warnings=[]
        supported=True
        for c in self.ir.components:
            ref,k=c["reference"],c["type"]
            ns=[p["node_id"] for p in c.get("pins",[])]
            p=c.get("parameters",{})
            try:
                if k=="V":
                    a,b=ns[:2]
                    spec=_source_wave(c)
                    if b==0 and a!=0: source_nodes[a]=spec
                    elif a==0 and b!=0:
                        source_nodes[b]=(lambda t,s=spec: -s(t) if callable(s) else -s)
                    else:
                        warnings.append(f"{ref}: ideal voltage source with neither terminal at ground is not supported by the nonlinear nodal backend."); supported=False
                    source_defs.append((ref,a,b,p))
                elif k=="R": model.add_resistor(*ns[:2],p["value"],name=ref)
                elif k=="C": model.add_capacitor(*ns[:2],p["value"],name=ref)
                elif k=="D":
                    mp=p.get("model_parameters",{})
                    if "BV" in mp:
                        model.add_zener(*ns[:2],zener_voltage=float(mp["BV"]),dynamic_resistance=float(mp.get("RBV",mp.get("RS",5.0))),name=ref)
                    else: model.add_diode(*ns[:2],saturation_current=float(p.get("IS",1e-14)),ideality=float(p.get("N",1.0)),name=ref)
                elif k=="Q":
                    typ=str(p.get("model_type") or "NPN").upper()
                    model.add_bjt(*ns[:3],beta=float(p.get("BF",100)),saturation_current=float(p.get("IS",1e-15)),transistor_type="PNP" if "PNP" in typ else "NPN",name=ref)
                elif k=="M":
                    typ=str(p.get("model_type") or "NMOS").upper()
                    inst=p.get("instance_parameters",{})
                    kp=float(p.get("KP",1e-3)); w=float(inst.get("W",p.get("W",1.0))); l=float(inst.get("L",p.get("L",1.0)))
                    kpar=kp*w/max(l,1e-30)
                    model.add_mosfet(ns[0],ns[1],ns[2],k=kpar,threshold=abs(float(p.get("VTO",1.0))),channel_length_modulation=float(p.get("LAMBDA",0.0)),mos_type="PMOS" if "PMOS" in typ else "NMOS",name=ref)
                    if len(ns)>3 and ns[3] not in (0,ns[2]): warnings.append(f"{ref}: MOSFET bulk node is not tied to source/ground; body effect is not modelled.")
                elif k=="J":
                    typ=str(p.get("model_type") or "NJF").upper()
                    model.add_jfet(ns[0],ns[1],ns[2],idss=float(p.get("BETA",5e-3)),pinch_off=float(p.get("VTO",-4.0)),transistor_type="PJF" if "PJF" in typ else "NJF",name=ref)
                elif k=="G": model.add_vccs(*ns[:4],float(p["gain"]),name=ref)
                elif k=="E":
                    rout=float(p.get("ROUT",10.0)); vlo=p.get("VLO"); vhi=p.get("VHI")
                    if ns[1]==0 and (vlo is not None or vhi is not None):
                        # A rail-limited E source is adapted to the preserved smooth
                        # OTA model, matching the legacy oscillator
                        # helper model more closely than a hard-clipped ideal VCVS.
                        model.add_ota(ns[0],ns[2],ns[3],gain=float(p["gain"]),
                                      output_resistance=rout,
                                      low_rail=float(vlo if vlo is not None else -1e9),
                                      high_rail=float(vhi if vhi is not None else 1e9),name=ref)
                    else:
                        model.add_vcvs(*ns[:4],float(p["gain"]),output_resistance=rout,
                                       low_rail=vlo,high_rail=vhi,name=ref)
                elif k=="S":
                    mp=p.get("model_parameters",{})
                    model.add_switch(*ns[:4],ron=float(mp.get("RON",1.0)),roff=float(mp.get("ROFF",1e9)),vt=float(mp.get("VT",0.5)),vh=float(mp.get("VH",0.0)),initial_closed=bool(p.get("initial_closed",False)),name=ref)
                else:
                    warnings.append(f"{ref}: element type {k} has no enabled nonlinear-backend adapter."); supported=False
            except Exception as exc:
                warnings.append(f"{ref}: adapter construction failed: {exc}"); supported=False
        if not source_nodes:
            warnings.append("No independent grounded voltage source was available to anchor the nonlinear nodal solver."); supported=False
        if len(source_nodes)!=len(set(source_nodes)):
            warnings.append("Multiple independent voltage sources share a fixed node."); supported=False
        model.sources=source_nodes
        return model, source_defs, warnings, supported

    def _mna_netlist(self):
        net={}; supported=True; why=[]
        for c in self.ir.components:
            k=c["type"]; ref=c["reference"]; ns=[p["node_id"] for p in c.get("pins",[])]; p=c.get("parameters",{})
            try:
                if k in "RCL": net[ref]=tuple(ns[:2]+[float(p["value"])])
                elif k in "VI": net[ref]=tuple(ns[:2]+[complex(p.get("dc",0.0),0.0)])
                elif k in "EG": net[ref]=tuple(ns[:4]+[float(p["gain"])])
                elif k=="Q":
                    # Only explicitly supplied linearized parameters can be accepted here.
                    gm=p.get("GM"); rpi=p.get("RPI"); ro=p.get("RO")
                    if None in (gm,rpi,ro): supported=False; why.append(f"{ref}: linear MNA Q model requires GM, RPI and RO.")
                    else: net[ref]=tuple(ns[:3]+[float(gm),float(rpi),float(ro)])
                else:
                    supported=False; why.append(f"{ref}: {k} is not supported by the linear MNA adapter.")
            except Exception as exc:
                supported=False; why.append(f"{ref}: MNA adapter failed: {exc}")
        return net,supported,why

    def _output_node(self):
        cfg=self.ir.analysis_configuration
        if "output_node" in cfg:
            name=str(cfg["output_node"]); return self._node_name_to_id.get(name,0)
        # Prefer a node named out/output, otherwise the highest non-ground ID.
        for name in ("out","output","vout","vo"):
            for raw,nid in self._node_name_to_id.items():
                if raw.casefold()==name:return nid
        return max(self._node_name_to_id.values(),default=0)

    def _output_pair(self):
        cfg=self.ir.analysis_configuration
        pos=cfg.get("output_node")
        neg=cfg.get("output_negative_node")
        if pos is None or neg is None: return None
        if str(pos) not in self._node_name_to_id or str(neg) not in self._node_name_to_id:
            raise ValueError(f"Differential output nodes {pos!r}, {neg!r} were not both found in the CircuitIR node table.")
        return (self._node_name_to_id[str(pos)], self._node_name_to_id[str(neg)])

    def _sweep_config(self, sources):
        cfg=self.ir.analysis_configuration.get("dc_sweep",{})
        if cfg.get("source"):
            source=cfg["source"]
            comp=next((c for c in self.ir.components if c["reference"].casefold()==source.casefold()),None)
            if comp: return source,float(cfg["start"]),float(cfg["stop"]),max(2,int(abs((float(cfg["stop"])-float(cfg["start"]))/float(cfg.get("step",0.1))))+1)
        if not sources:return None
        c=sources[0]; p=c["parameters"]; base=float(p.get("dc",0.0)); span=max(abs(base),1.0)
        return c["reference"],0.0,max(1.0,1.25*span),41

    def _analog_metrics(self, model, op, source_defs):
        volts={int(k):float(v) for k,v in op.get("node_voltages_V",{}).items()}
        residual={n:0.0 for n in self._id_to_name if n!=0}
        components={}; source_nodes=set(model.sources); engineering_warnings=[]
        for idx,e in enumerate(model.elements):
            name=e.get("name") or f"{e['kind']}{idx+1}"
            try: currents=np.asarray(model._element_currents(idx,e,volts,states=model._schmitt_state),dtype=float)
            except Exception: continue
            pins=e["nodes"]; pin_values=[]
            for pin,cur in zip(pins,currents):
                vp=volts.get(pin,0.0); pin_values.append({"node":self._id_to_name.get(pin,str(pin)),"voltage_V":vp,"current_leaving_A":float(cur)})
                if pin!=0: residual[pin]=residual.get(pin,0.0)+float(cur)
            power=sum(volts.get(pin,0.0)*float(cur) for pin,cur in zip(pins,currents))
            terminals=e["nodes"]
            comp_ir=next((c for c in self.ir.components if c["reference"]==name),None)
            item={"terminal_values":pin_values,"power_W":power}
            if len(terminals)>=2: item["voltage_V"]=volts.get(terminals[0],0.0)-volts.get(terminals[1],0.0); item["current_A"]=float(currents[0])
            kind=e["kind"]
            if kind=="R": item["resistance_ohm"]=e["value"]
            if kind=="C":
                v=item.get("voltage_V",0.0); item.update({"capacitance_F":e["value"],"charge_C":e["value"]*v,"stored_energy_J":0.5*e["value"]*v*v})
            if kind=="D": item["operating_state"]="forward_conducting" if item.get("voltage_V",0)>0.25 else "reverse_or_cutoff"
            if kind=="Z": item["operating_state"]="zener_breakdown" if item.get("voltage_V",0)<-e["vz"] else "below_breakdown_or_forward"
            if kind=="Q":
                c,b,em=terminals; pol=1 if e["transistor_type"]=="NPN" else -1
                vbe=pol*(volts.get(b,0)-volts.get(em,0)); vce=pol*(volts.get(c,0)-volts.get(em,0))
                item.update({"VBE_V":vbe,"VCE_V":vce,"base_current_A":float(currents[1]),"collector_current_A":float(currents[0]),"emitter_current_A":float(currents[2]),"operating_region":"cutoff" if vbe<0.55 else "saturation" if vce<0.2 else "forward_active"})
            if kind=="M":
                d,g,src=terminals; pol=1 if e["mos_type"]=="NMOS" else -1
                vgs=pol*(volts.get(g,0)-volts.get(src,0)); vds=pol*(volts.get(d,0)-volts.get(src,0)); vov=vgs-e["vt"]
                item.update({"VGS_V":vgs,"VDS_V":vds,"transconductance_approx_S":e["k"]*max(vov,0.0),"operating_region":"cutoff" if vov<=0 else "linear" if abs(vds)<vov else "saturation"})
            if kind=="J":
                d,g,src=terminals; pol=1 if e["jfet_type"]=="NJF" else -1
                vgs=pol*(volts.get(g,0)-volts.get(src,0)); vds=pol*(volts.get(d,0)-volts.get(src,0)); vp=e["vp"]
                gm=2*e["idss"]/max(abs(vp),1e-30)*max(0.0,1-vgs/vp)
                item.update({"VGS_V":vgs,"VDS_V":vds,"transconductance_approx_S":gm,"operating_region":"cutoff" if vgs<=vp else "ohmic" if abs(vds)<abs(vgs-vp) else "pinch_off"})
                if (e["jfet_type"]=="NJF" and vds < -1e-9) or (e["jfet_type"]=="PJF" and vds < -1e-9):
                    item["validation_status"]="physically_inconsistent_drain_source_polarity"
                    warning={"code":"JFET_NEGATIVE_VDS","component":name,
                             "message":f"{e['jfet_type']} device has polarity-normalized VDS={vds:.9g} V. The assumed drain/source orientation is inconsistent with the usual forward drain-bias operating convention; review the original bias assumptions and terminal order."}
                    item["validation_warning"]=warning["message"]
                    engineering_warnings.append(warning)
                else:
                    item["validation_status"]="polarity_consistent_with_forward_drain_bias"
            components[name]=item
        # Infer current through each ideal voltage source using the KCL balance at its fixed node.
        source_results={}
        for ref,a,b,p in source_defs:
            va=volts.get(a,0.0); vb=volts.get(b,0.0)
            if a!=0: ia=-residual.get(a,0.0)
            elif b!=0: ia=residual.get(b,0.0)
            else: ia=0.0
            source_results[ref]={"voltage_V":va-vb,"current_A":ia,"power_W":(va-vb)*ia}
            for node,cur in ((a,ia),(b,-ia)):
                if node!=0: residual[node]=residual.get(node,0.0)+cur
        max_kcl=max((abs(v) for n,v in residual.items() if n not in source_nodes),default=0.0)
        return {"components":components,"voltage_sources":source_results,"engineering_warnings":engineering_warnings,"kcl":{"node_residuals_A":{self._id_to_name.get(n,str(n)):v for n,v in residual.items()},"max_residual_A":max_kcl,"is_balanced":bool(max_kcl<1e-6)},"power_balance":{"sum_power_W":sum(v["power_W"] for v in components.values())+sum(v["power_W"] for v in source_results.values()),"status":"calculated_with_model_terminal_currents"}}

    @staticmethod
    def _response_metrics(results):
        metrics={}
        ac=results.get("ac_sweep",{})
        if ac.get("status")=="success" and ac.get("data"):
            d=ac["data"]; f=d.get("frequency_Hz",d.get("frequency")); mag=d.get("magnitude_dB")
            if mag is None and d.get("magnitude") is not None: mag=20*np.log10(np.maximum(np.asarray(d["magnitude"]),1e-300))
            if f is not None and mag is not None and len(f):
                f=np.asarray(f,dtype=float); mag=np.asarray(mag,dtype=float); imax=int(np.argmax(mag)); peak=float(mag[imax]); target=peak-3.0103
                crossings=np.where(np.diff((mag>=target).astype(int))!=0)[0]
                metrics["ac_response"]={"peak_frequency_Hz":float(f[imax]),"peak_gain_dB":peak,"minus_3dB_level_dB":target,"crossing_frequencies_Hz":[float(f[i]) for i in crossings],"phase_available":bool(d.get("phase_deg") is not None or d.get("phase") is not None)}
        tr=results.get("transient",{})
        if tr.get("status")=="success" and tr.get("data"):
            d=tr["data"]; y=d.get("output_V")
            if y is None and d.get("outputs") is not None:
                a=np.asarray(d["outputs"]); y=a[:,0] if a.ndim>1 else a
            if y is not None:
                y=np.asarray(y,dtype=float); metrics["transient_output"]={"minimum_V":float(np.min(y)),"maximum_V":float(np.max(y)),"peak_to_peak_V":float(np.ptp(y)),"rms_V":float(np.sqrt(np.mean(y*y))),"mean_V":float(np.mean(y))}
                if len(y)>2:
                    z=y-np.mean(y); crossings=np.where(np.diff(np.signbit(z)))[0]
                    if len(crossings)>2 and d.get("time_s") is not None:
                        t=np.asarray(d["time_s"]); periods=np.diff(t[crossings]); metrics["transient_output"]["estimated_frequency_Hz"]=float(1/(2*np.mean(periods))) if np.mean(periods)>0 else None
        return metrics

    def execute(self, analyses="all"):
        requested=set(self.ANALYSIS_KEYS if analyses=="all" else ([analyses] if isinstance(analyses,str) else analyses))
        results={k:{"status":"skipped","reason":"Not requested.","data":None} for k in self.ANALYSIS_KEYS}
        modules=[]; numerical={}; engineering={}; warnings=list(self.ir.validation_results.get("warnings",[])); errors=list(self.ir.validation_results.get("errors",[]))
        # Determine adapter fit. Nonlinear nodal backend takes priority for nonlinear devices.
        analog_model, analog_sources, analog_warnings, analog_ok=self._analog_model()
        mna_net,mna_ok,mna_warnings=self._mna_netlist()
        sparse_mna_ok,sparse_mna_warnings=supports_circuit_ir_dc(self.ir)
        nonlinear_mna_ok, nonlinear_mna_warnings = supports_circuit_ir_nonlinear_dc(self.ir)
        transient_mna_ok, transient_mna_warnings = supports_circuit_ir_transient(self.ir)
        nonlinear_transient_ok, nonlinear_transient_warnings = supports_circuit_ir_nonlinear_transient(self.ir)
        nonlinear=any(
            c["type"] in ("D","Q","J","M","S")
            or (c["type"]=="E" and any(key in c.get("parameters",{}) for key in ("ROUT", "VLO", "VHI")))
            for c in self.ir.components
        )
        # Parameter keys outside the supported linear-DC subset may only fall
        # through when a known analog adapter explicitly handles them. Never
        # allow an unknown instance/model parameter to disappear silently.
        unhandled_parameter_refs=[]
        for component in self.ir.components:
            ref=str(component.get("reference", "?")); kind=str(component.get("type", "?")).upper()
            reasons=[reason for reason in sparse_mna_warnings if reason.startswith(ref + ": unsupported ") and "parameter(s)" in reason]
            if reasons:
                extra_keys=set(component.get("parameters", {})) - {
                    "R": {"value", "raw_value", "unit"}, "C": {"value", "raw_value", "unit", "IC"},
                    "L": {"value", "raw_value", "unit", "IC"},
                    "V": {"dc", "ac_magnitude", "ac_phase_deg", "ac_specified", "waveform"},
                    "I": {"dc", "ac_magnitude", "ac_phase_deg", "ac_specified", "waveform"},
                    "E": {"gain", "raw_value"}, "G": {"gain", "raw_value"},
                    "F": {"control_source", "gain"}, "H": {"control_source", "gain"},
                }.get(kind, set())
                if not (kind == "E" and extra_keys.issubset({"ROUT", "VLO", "VHI"})):
                    unhandled_parameter_refs.append((ref, reasons))
        if unhandled_parameter_refs:
            backend="unsupported"
        elif requested == {"transient"} and (transient_mna_ok or nonlinear_transient_ok):
            backend="mna_transient"
        elif requested == {"transient"} and any(
            c.get("type") in ("D", "Q", "J", "M") for c in self.ir.components
        ) and not nonlinear_transient_ok:
            # Do not silently fall back to the older approximate analog
            # transient adapter for semiconductor MNA circuits.
            backend="unsupported"
        elif requested == {"dc_operating_point"} and nonlinear_mna_ok:
            # Preserve the original backend identifier for callers that consume
            # execution_summary; the implementation is now generalized D/Q/J/M.
            backend="nonlinear_mna_dc"
        elif requested and "ac_sweep" in requested and requested.issubset({"dc_operating_point", "ac_sweep"}) and nonlinear_mna_ok:
            backend="nonlinear_mna"
        elif nonlinear and analog_ok:
            backend="nonlinear_analog"
        elif sparse_mna_ok:
            backend="sparse_mna_dc"
        elif mna_ok:
            backend="linear_mna"
        elif analog_ok:
            backend="nonlinear_analog"
        else:
            backend="unsupported"
        # Adapter-fit diagnostics are relevant only when this backend is chosen.
        # Reporting failed attempts to build a secondary backend created false
        # warnings for valid linear circuits (for example an RLC network).
        if backend == "nonlinear_analog":
            warnings.extend({"code":"ADAPTER_WARNING","message":w} for w in analog_warnings)
        output=self._output_node()
        output_pair=self._output_pair()
        source_components=[c for c in self.ir.components if c["type"]=="V"]
        src_cfg=self._sweep_config(source_components)
        has_storage=any(c["type"] in ("C","L") and float(c.get("parameters",{}).get("value",0))>0 for c in self.ir.components)
        has_c=any(c["type"]=="C" for c in self.ir.components)
        has_l=any(c["type"]=="L" for c in self.ir.components)
        try:
            if backend == "mna_transient":
                try:
                    transient_config = self.ir.analysis_configuration.get("transient", {})
                    configured_method = str(self.ir.simulation_options.get("METHOD", "backward_euler")).lower()
                    method = "trapezoidal" if configured_method in ("trap", "trapezoidal") else "backward_euler"
                    options = self.ir.simulation_options
                    adaptive_value = options.get("ADAPTIVE", transient_config.get("adaptive", 0))
                    tr = solve_mna_transient(
                        self.ir, output_node=output, output_pair=output_pair,
                        method=method,
                        adaptive=bool(float(adaptive_value)),
                        relative_tolerance=float(options.get("RELTOL", 1e-3)),
                        absolute_voltage_tolerance=float(options.get("VABSTOL", options.get("ABSTOL", 1e-6))),
                        absolute_current_tolerance=float(options.get("IABSTOL", 1e-9)),
                        gmin_S=float(options.get("GMIN", 1e-12)),
                        minimum_step_s=(float(options["TMIN"]) if options.get("TMIN") is not None else None),
                        maximum_steps=int(float(options.get("MAXSTEPS", 200000))),
                        maximum_newton_iterations=int(float(options.get("ITL1", 60))),
                    ).data
                    results["transient"] = {"status": "success", "reason": None, "data": tr}
                    modules.append({"module": "circhecker.solver.transient", "analysis": "transient",
                                    "status": "success", "result_reference": "analysis_results.transient",
                                    "diagnostics": tr.get("solver_diagnostics")})
                    numerical["node_voltages_V"] = {name: values[-1] for name, values in tr["node_voltages_V"].items()}
                except Exception as exc:
                    results["transient"] = {"status": "failed", "reason": str(exc), "data": None}
                    modules.append({"module": "circhecker.solver.transient", "analysis": "transient",
                                    "status": "failed", "error": str(exc)})
            elif backend in ("nonlinear_mna", "nonlinear_mna_dc"):
                op_solution = None
                try:
                    op_solution = solve_nonlinear_dc(self.ir)
                    op = op_solution.to_dict()
                    if "dc_operating_point" in requested:
                        results["dc_operating_point"] = {"status": "success", "reason": None, "data": op}
                        modules.append({"module": "circhecker.solver.nonlinear", "analysis": "dc_operating_point",
                                        "status": "success", "result_reference": "analysis_results.dc_operating_point",
                                        "diagnostics": op.get("solver_diagnostics")})
                        numerical["node_voltages_V"] = {self._id_to_name.get(int(n), str(n)): float(v)
                                                         for n, v in op["node_voltages_V"].items()}
                        numerical["branch_results"] = op["branch_results"]
                        numerical["device_results"] = op["device_results"]
                        engineering["kcl"] = op["verification"]["kcl"]
                        engineering["newton_residual"] = op["verification"]["newton_residual"]
                except Exception as exc:
                    if "dc_operating_point" in requested:
                        results["dc_operating_point"] = {"status": "failed", "reason": str(exc), "data": None}
                        modules.append({"module": "circhecker.solver.nonlinear", "analysis": "dc_operating_point",
                                        "status": "failed", "error": str(exc)})
                    if "ac_sweep" in requested:
                        results["ac_sweep"] = {"status": "failed", "reason": f"Nonlinear DC operating point required for AC failed: {exc}", "data": None}
                        modules.append({"module": "circhecker.solver.nonlinear_ac", "analysis": "ac_sweep",
                                        "status": "failed", "error": str(exc)})
                if "ac_sweep" in requested and op_solution is not None:
                    try:
                        ac_cfg = self.ir.analysis_configuration.get("ac_sweep", {})
                        frequencies = frequency_sweep(ac_cfg or None)
                        ac = solve_nonlinear_ac(self.ir, operating_point=op_solution, frequencies_hz=frequencies)
                        positive_node = output_pair[0] if output_pair is not None else output
                        negative_node = output_pair[1] if output_pair is not None else 0
                        positive_name = self._id_to_name.get(positive_node, str(positive_node))
                        negative_name = self._id_to_name.get(negative_node, str(negative_node))
                        vp = ac["node_voltages_V"].get(positive_name, [None] * len(frequencies))
                        vn = ac["node_voltages_V"].get(negative_name, [None] * len(frequencies))
                        output_values = [((a - b) if a is not None and b is not None else None)
                                         for a, b in zip(vp, vn)]
                        magnitudes = [abs(value) if value is not None else None for value in output_values]
                        ac["output_nodes"] = [positive_name, negative_name]
                        ac["output_voltage_V"] = output_values
                        ac["magnitude"] = magnitudes
                        ac["magnitude_dB"] = [20.0 * math.log10(max(value, 1e-300)) if value is not None else None
                                               for value in magnitudes]
                        ac["phase_deg"] = [float(np.degrees(np.angle(value))) if value is not None else None
                                            for value in output_values]
                        ac["interpretation"] = "Small-signal nonlinear MNA about the converged DC operating point; device admittance is di/dv + j*omega*dq/dv."
                        ac_status = "failed" if ac["solver_diagnostics"]["failed_frequency_count"] else "success"
                        ac_reason = (f"{ac['solver_diagnostics']['failed_frequency_count']} AC frequency point(s) failed."
                                     if ac_status == "failed" else None)
                        results["ac_sweep"] = {"status": ac_status, "reason": ac_reason, "data": ac}
                        modules.append({"module": "circhecker.solver.nonlinear_ac", "analysis": "ac_sweep",
                                        "status": ac_status, "result_reference": "analysis_results.ac_sweep",
                                        "diagnostics": ac.get("solver_diagnostics")})
                        if ac_status == "success":
                            numerical["ac_output_voltage_V"] = output_values
                    except Exception as exc:
                        results["ac_sweep"] = {"status": "failed", "reason": str(exc), "data": None}
                        modules.append({"module": "circhecker.solver.nonlinear_ac", "analysis": "ac_sweep",
                                        "status": "failed", "error": str(exc)})
            elif backend=="nonlinear_analog":
                # Run OP
                if "dc_operating_point" in requested:
                    try:
                        ic={self._node_name_to_id.get(str(k),0):v for k,v in self.ir.initial_conditions.items()}
                        op=analog_model.dc_operating_point(initial_conditions=ic,plot=False)
                        results["dc_operating_point"]={"status":"success","reason":None,"data":op}
                        modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"dc_operating_point","status":"success","result_reference":"analysis_results.dc_operating_point"})
                        numerical["node_voltages_V"]={self._id_to_name.get(int(n),str(n)):float(v) for n,v in op["node_voltages_V"].items()}
                        numerical["element_currents_A"]=op["branch_currents_A"]
                        numerical["component_electrical_metrics"]=self._analog_metrics(analog_model,op,analog_sources)
                        warnings.extend(numerical["component_electrical_metrics"].get("engineering_warnings",[]))
                        engineering["kcl"]=numerical["component_electrical_metrics"]["kcl"]
                        engineering["power_balance"]=numerical["component_electrical_metrics"]["power_balance"]
                    except Exception as exc:
                        results["dc_operating_point"]={"status":"failed","reason":str(exc),"data":None}; modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"dc_operating_point","status":"failed","error":str(exc)})
                else: results["dc_operating_point"]={"status":"skipped","reason":"Not requested.","data":None}
                # DC sweep
                if "dc_sweep" in requested and src_cfg:
                    srcname,st,sp,pts=src_cfg
                    source=next(c for c in source_components if c["reference"].casefold()==srcname.casefold())
                    p=source["parameters"]; ns=[x["node_id"] for x in source["pins"]]
                    weight=1.0 if ns[0]!=0 else -1.0
                    srcnode=ns[0] if ns[0]!=0 else ns[1]
                    if ns[0]==0: st,sp=-sp,-st
                    try:
                        sw=analog_model.dc_sweep(st,sp,pts,source_weights={srcnode:weight},output_node=output,output_pair=output_pair,plot=False)
                        results["dc_sweep"]={"status":"success","reason":None,"data":sw}; modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"dc_sweep","status":"success","result_reference":"analysis_results.dc_sweep"})
                    except Exception as exc:
                        results["dc_sweep"]={"status":"failed","reason":str(exc),"data":None}; modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"dc_sweep","status":"failed","error":str(exc)})
                elif "dc_sweep" in requested: results["dc_sweep"]={"status":"unsupported","reason":"The nonlinear nodal backend requires a grounded independent voltage source for DC sweep.","data":None}
                # AC sweep: explicitly a local small-signal linearization.
                ac_cfg=self.ir.analysis_configuration.get("ac_sweep",{})
                custom_ac_input=self.ir.analysis_configuration.get("ac_input_node")
                if "ac_sweep" in requested and (source_components or custom_ac_input):
                    f0=float(ac_cfg.get("start_Hz",1.0)); f1=float(ac_cfg.get("stop_Hz",1e6)); points=min(500,max(16,int(ac_cfg.get("points",160))))
                    if custom_ac_input:
                        srcnode=self._node_name_to_id.get(str(custom_ac_input))
                        if srcnode is None: raise ValueError(f".circhecker AC_INPUT node {custom_ac_input!r} does not exist.")
                        weight=complex(1.0,0.0)
                        ac_output=self._node_name_to_id.get(str(self.ir.analysis_configuration.get("ac_output_node",self._id_to_name.get(output,"out"))),output)
                    else:
                        ac_source=next((c for c in source_components if c["parameters"].get("ac_specified") or c["parameters"].get("waveform")),source_components[0])
                        ns=[x["node_id"] for x in ac_source["pins"]]
                        acmag=float(ac_source["parameters"].get("ac_magnitude",1.0)); phase=math.radians(float(ac_source["parameters"].get("ac_phase_deg",0.0)))
                        if acmag==0: acmag=1.0
                        srcnode=ns[0] if ns[0]!=0 else ns[1]; weight=(1 if ns[0]!=0 else -1)*acmag*np.exp(1j*phase)
                        ac_output=output
                    try:
                        ac_ics={self._node_name_to_id.get(str(k),0):v for k,v in self.ir.initial_conditions.items()}
                        ac=analog_model.ac_sweep(
                            f0, f1, points, source_weights={srcnode:weight},
                            output_node=ac_output, output_pair=output_pair, plot=False,
                            dc_initial_conditions=ac_ics,
                        )
                        ac["interpretation"]="Small-signal AC linearization about the computed DC operating point; not a Fourier spectrum of nonlinear periodic waveforms."
                        results["ac_sweep"]={"status":"success","reason":None,"data":ac}; modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"ac_sweep","status":"success","result_reference":"analysis_results.ac_sweep"})
                    except Exception as exc:
                        results["ac_sweep"]={"status":"failed","reason":str(exc),"data":None}; modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"ac_sweep","status":"failed","error":str(exc)})
                elif "ac_sweep" in requested: results["ac_sweep"]={"status":"unsupported","reason":"No independent voltage excitation or .circhecker AC_INPUT injection was available.","data":None}
                # Transient only with C; no ideal inductor in this backend.
                has_waveform=any(c["parameters"].get("waveform") for c in source_components)
                unsupported_waveforms=[(c["reference"],c["parameters"]["waveform"]["kind"]) for c in source_components
                                       if c["parameters"].get("waveform") and c["parameters"]["waveform"].get("kind") not in ("SIN","PULSE","PWL")]
                if "transient" in requested and unsupported_waveforms:
                    results["transient"]={"status":"unsupported","reason":f"Unsupported time-domain source waveform(s): {unsupported_waveforms}.","data":None}
                elif "transient" in requested and (has_c or has_waveform) and not has_l and source_components:
                    tc=self.ir.analysis_configuration.get("transient",{})
                    stop=float(tc.get("stop_s",0.02)); step=float(tc.get("step_s",stop/400 if stop>0 else 1e-5)); n=int(min(6000,max(40,math.ceil(stop/max(step,1e-15))+1)))
                    time=np.linspace(0.0,stop,n)
                    sig={}
                    for s in source_components:
                        ns=[p["node_id"] for p in s["pins"]]; node=ns[0] if ns[0]!=0 else ns[1]
                        if node!=0:
                            wf=_source_wave(s); sig[node]=wf if callable(wf) else (lambda t,v=wf:v)
                    try:
                        ics={self._node_name_to_id.get(str(k),0):v for k,v in self.ir.initial_conditions.items()}
                        tr=analog_model.transient(time,source_signals=sig,initial_conditions=ics,output_node=output,output_pair=output_pair,plot=False)
                        results["transient"]={"status":"success","reason":None,"data":tr}; modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"transient","status":"success","result_reference":"analysis_results.transient"})
                    except Exception as exc:
                        results["transient"]={"status":"failed","reason":str(exc),"data":None}; modules.append({"module":"circhecker.solver.analog.analogcircuit","analysis":"transient","status":"failed","error":str(exc)})
                elif "transient" in requested:
                    reason="No energy-storage element or time-varying independent source is present." if not has_storage and not has_waveform else "The nonlinear transient adapter does not yet model ideal inductor states."
                    results["transient"]={"status":"not_applicable" if not has_storage and not has_waveform else "unsupported","reason":reason,"data":None}
                # nonlinear frequency / TF/state-space
                for k in ("frequency_analysis","transfer_function","state_space"):
                    if k in requested: results[k]={"status":"not_applicable","reason":"A linearized local frequency response is reported in ac_sweep; descriptor state-space extraction for this nonlinear model is not exposed.","data":None}
            elif backend in ("linear_mna", "sparse_mna_dc"):
                # The new sparse MNA implementation is authoritative for supported
                # linear DC operating points. Legacy AC/transfer paths remain in place
                # during this incremental migration when their adapter accepts the IR.
                mna=circuitsolver(mna_net) if mna_ok else None
                if backend == "sparse_mna_dc" and not mna_ok:
                    warnings.extend({"code":"LEGACY_LINEAR_ADAPTER_UNAVAILABLE","message":w} for w in mna_warnings)
                if "dc_operating_point" in requested:
                    try:
                        if backend == "sparse_mna_dc":
                            op=solve_sparse_mna_dc(self.ir).to_dict()
                        else:
                            op=mna.solve_dc(); op["verification"]=mna.verify_dc()
                        dc_module = "circhecker.solver.mna" if backend == "sparse_mna_dc" else "circhecker.solver.linear.circuitsolver"
                        results["dc_operating_point"]={"status":"success","reason":None,"data":op}; modules.append({"module":dc_module,"analysis":"dc_operating_point","status":"success","result_reference":"analysis_results.dc_operating_point","diagnostics":op.get("solver_diagnostics")})
                        numerical["node_voltages_V"]={self._id_to_name.get(int(n),str(n)):float(np.real(v)) for n,v in op["node_voltages_V"].items()}
                        numerical["branch_results"]=op["branch_results"]
                        engineering["kcl"]=op["verification"]["kcl"]
                        engineering["kvl"]=op["verification"].get("kvl",[])
                        signed_power=sum(float(np.real(v.get("power_W",0.0))) for v in op["branch_results"].values())
                        power_scale=sum(abs(float(np.real(v.get("power_W",0.0)))) for v in op["branch_results"].values())
                        engineering["power_balance"]={
                            "total_signed_power_W":signed_power,
                            "absolute_power_sum_W":power_scale,
                            "is_balanced":bool(abs(signed_power)<=max(1e-9,1e-6*power_scale)),
                            "status":"calculated_from_MNA_branch_results",
                        }
                    except Exception as exc:
                        results["dc_operating_point"]={"status":"failed","reason":str(exc),"data":None}; modules.append({"module":"circhecker.solver.linear.circuitsolver","analysis":"dc_operating_point","status":"failed","error":str(exc)})
                # transferfunction needs one selected input; build using first V or I source.
                srcs=[c for c in self.ir.components if c["type"] in "VI"]
                tf=None
                if srcs and mna is not None:
                    try: tf=transferfunction(mna,[srcs[0]["reference"]],[output])
                    except Exception as exc: warnings.append({"code":"TRANSFER_FUNCTION_INIT","message":str(exc)})
                if "dc_sweep" in requested and tf is not None and src_cfg:
                    srcname,st,sp,pts=src_cfg
                    try:
                        sw=tf.dc_sweep(srcname,st,sp,pts,output=output,plot=False)
                        results["dc_sweep"]={"status":"success","reason":None,"data":sw}; modules.append({"module":"circhecker.solver.transfer_function.transferfunction","analysis":"dc_sweep","status":"success","result_reference":"analysis_results.dc_sweep"})
                    except Exception as exc: results["dc_sweep"]={"status":"failed","reason":str(exc),"data":None}
                elif "dc_sweep" in requested: results["dc_sweep"]={"status":"unsupported","reason":"No compatible legacy sweep adapter or independent source was available.","data":None}
                if "ac_sweep" in requested:
                    ac_cfg=self.ir.analysis_configuration.get("ac_sweep",{})
                    ac_supported, ac_reasons = supports_circuit_ir_ac(self.ir)
                    if ac_supported:
                        try:
                            frequencies=frequency_sweep(ac_cfg or None)
                            ac_pair=output_pair if output_pair is not None else (output,0)
                            ac_solution=solve_sparse_mna_ac(
                                self.ir, frequencies_hz=frequencies, output_nodes=ac_pair
                            )
                            ac=ac_solution.data
                            ac["interpretation"]="Complex sparse linear AC MNA response; source phasors follow explicit AC cards or the documented implicit unit-input convention."
                            ac_result=ac_solution.to_dict()
                            results["ac_sweep"]={"status":ac_result["status"],"reason":ac_result["reason"],"data":ac}
                            modules.append({"module":"circhecker.solver.ac","analysis":"ac_sweep",
                                            "status":ac_result["status"],"result_reference":"analysis_results.ac_sweep",
                                            "diagnostics":ac.get("solver_diagnostics")})
                            if ac_result["status"]!="success":
                                errors.append({"code":"AC_MNA_FREQUENCY_FAILURE","message":ac_result["reason"],
                                               "failed_frequencies_Hz":ac["solver_diagnostics"].get("failed_frequencies_Hz",[])})
                        except Exception as exc:
                            results["ac_sweep"]={"status":"failed","reason":str(exc),"data":None}
                            modules.append({"module":"circhecker.solver.ac","analysis":"ac_sweep","status":"failed","error":str(exc)})
                    elif tf is not None:
                        # Temporary compatibility fallback for linear topologies outside
                        # the generalized AC element subset. The selected backend is
                        # recorded so this cannot be mistaken for generalized MNA.
                        try:
                            ac=tf.ac_sweep(float(ac_cfg.get("start_Hz",1)),float(ac_cfg.get("stop_Hz",1e6)),min(500,max(16,int(ac_cfg.get("points",160)))),plot=False)
                            ac["interpretation"]="Legacy transfer-function AC response; topology is outside generalized AC MNA scope."
                            results["ac_sweep"]={"status":"success","reason":None,"data":ac}
                            modules.append({"module":"circhecker.solver.transfer_function.transferfunction","analysis":"ac_sweep","status":"success","result_reference":"analysis_results.ac_sweep","fallback_reasons":ac_reasons})
                        except Exception as exc:
                            results["ac_sweep"]={"status":"failed","reason":str(exc),"data":None}
                    else:
                        results["ac_sweep"]={"status":"unsupported","reason":"Generalized AC MNA cannot represent this topology: "+"; ".join(ac_reasons),"data":None}
                unsupported_waveforms=[(c["reference"],c["parameters"]["waveform"]["kind"]) for c in srcs
                                       if c["parameters"].get("waveform") and c["parameters"]["waveform"].get("kind") not in ("SIN","PULSE","PWL")]
                if "transient" in requested and unsupported_waveforms:
                    results["transient"]={"status":"unsupported","reason":f"Unsupported time-domain source waveform(s): {unsupported_waveforms}.","data":None}
                elif "transient" in requested and tf is not None and has_storage:
                    tc=self.ir.analysis_configuration.get("transient",{}); stop=float(tc.get("stop_s",0.02)); step=float(tc.get("step_s",stop/400)); n=int(min(1200,max(40,math.ceil(stop/max(step,1e-15))+1))); time=np.linspace(0,stop,n)
                    waveform=_source_wave(srcs[0]);
                    if not callable(waveform): waveform=(lambda t,v=waveform:v)
                    try:
                        tr=tf.transient(time,waveform,plot=False,check_consistency=False)
                        # Preserve the output node identity so signal analysis can
                        # safely map the legacy single-output trace back to CircuitIR.
                        tr["output_node"] = self._id_to_name.get(output, str(output))
                        results["transient"]={"status":"success","reason":None,"data":tr}; modules.append({"module":"circhecker.solver.transfer_function.transferfunction","analysis":"transient","status":"success","result_reference":"analysis_results.transient"})
                    except Exception as exc: results["transient"]={"status":"failed","reason":str(exc),"data":None}
                elif "transient" in requested: results["transient"]={"status":"not_applicable" if not has_storage else "unsupported","reason":"No capacitor or inductor energy storage is present." if not has_storage else "No valid linear state-space input/output model was constructed.","data":None}
                if "state_space" in requested:
                    try:
                        if mna is None:
                            raise NotImplementedError("State-space extraction remains unavailable for circuits containing elements outside the legacy descriptor adapter.")
                        ss=mna.state_space()
                        results["state_space"]={"status":"success","reason":None,"data":ss}; modules.append({"module":"circhecker.solver.linear.circuitsolver","analysis":"state_space","status":"success","result_reference":"analysis_results.state_space"})
                    except Exception as exc: results["state_space"]={"status":"unsupported" if isinstance(exc, NotImplementedError) else "failed","reason":str(exc),"data":None}
                for k in ("frequency_analysis","transfer_function"):
                    if k in requested:
                        if tf is not None:
                            try:
                                data={"dc_gain":tf.dc_gain(),"poles":tf.poles(),"stability":tf.stability(),"diagnostics":tf.diagnostics()}
                                results[k]={"status":"success","reason":None,"data":data}
                            except Exception as exc: results[k]={"status":"unsupported","reason":str(exc),"data":None}
                        else: results[k]={"status":"unsupported","reason":"Linear transfer function unavailable.","data":None}
            else:
                for k in requested:
                    results[k]={"status":"unsupported","reason":"The parsed topology contains elements without a compatible execution adapter. See component validation and warnings.","data":None}
                warnings.extend({"code":"MNA_ADAPTER","message":w} for w in mna_warnings)
                warnings.extend({"code":"SPARSE_MNA_ADAPTER","message":w} for w in sparse_mna_warnings)
                if requested == {"transient"} and any(c.get("type") in ("D", "Q", "J", "M") for c in self.ir.components):
                    warnings.extend({"code":"NONLINEAR_TRANSIENT_ADAPTER","message":w} for w in nonlinear_transient_warnings)
                warnings.extend({"code":"BACKEND_UNAVAILABLE","message":w} for w in analog_warnings)
        except Exception as exc:
            errors.append({"code":"ORCHESTRATOR_ERROR","message":str(exc),"traceback":traceback.format_exc(limit=3)})
        numerical.update(self._response_metrics(results))
        # Meaningful no-storage transient status for resistive-only circuits.
        for k in self.ANALYSIS_KEYS:
            if k not in requested: results[k]={"status":"skipped","reason":"Not requested.","data":None}
            elif results[k]["status"]=="skipped": results[k]={"status":"unsupported","reason":"No compatible backend claimed this analysis.","data":None}
        engineering.setdefault("topology",{"node_count":len(self.ir.nodes)-1,"component_count":len(self.ir.components),"branch_count":len(self.ir.branches)})
        engineering.setdefault("power_balance",{"status":"not_calculated","reason":"A complete signed terminal-power sum is not available for every supported model family."})
        # Preserve a complete per-analysis execution ledger, including unavailable
        # capabilities. This prevents missing modules from being mistaken for success.
        module_name={"mna_transient":"circhecker.solver.transient + nonlinear_transient (linear RLC and nonlinear charge/flux MNA)",
                     "nonlinear_mna":"circhecker.solver.nonlinear (unified D/Q/J/M Newton DC + small-signal AC)",
                     "nonlinear_mna_dc":"circhecker.solver.nonlinear (compatibility alias)",
                     "sparse_mna_dc":"circhecker.solver.mna (DC) + legacy linear AC adapters where compatible",
                     "linear_mna":"circhecker.solver.linear.circuitsolver / circhecker.solver.transfer_function.transferfunction",
                     "nonlinear_analog":"circhecker.solver.analog.analogcircuit"}.get(backend,"circhecker.execution.engine")
        config_keys={"dc_operating_point":"dc_operating_point","dc_sweep":"dc_sweep","ac_sweep":"ac_sweep","transient":"transient"}
        for analysis in self.ANALYSIS_KEYS:
            if analysis not in requested: continue
            if not any(item.get("analysis")==analysis for item in modules):
                state=results[analysis]
                record={"module":module_name,"analysis":analysis,"status":state["status"],
                        "input_configuration":self.ir.analysis_configuration.get(config_keys.get(analysis,""),{"requested":True}),
                        "diagnostics":state.get("reason")}
                if state["status"]=="success": record["result_reference"]=f"analysis_results.{analysis}"
                if state["status"]=="failed": record["execution_error"]=state.get("reason")
                modules.append(record)
        for item in modules:
            item.setdefault("input_configuration",self.ir.analysis_configuration.get(config_keys.get(item.get("analysis"),""),{"requested":True}))
            item.setdefault("diagnostics",results.get(item.get("analysis"),{}).get("reason"))
        self.ir.validation_results.update({"warnings":warnings,"errors":errors})
        return {"analysis_results":results,"module_execution":modules,"numerical_metrics":numerical,"engineering_checks":engineering,"warnings":warnings,"errors":errors,"execution_summary":{"backend":backend,"requested_analyses":sorted(requested),"successful_analyses":[k for k,v in results.items() if v["status"]=="success"],"failed_analyses":[k for k,v in results.items() if v["status"]=="failed"],"unsupported_analyses":[k for k,v in results.items() if v["status"] in ("unsupported","not_applicable")],"capability_profile":{"component_registry":self.component_registry.profile(),"solver_registry":self.solver_registry.profile()}}}
