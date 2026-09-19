"""Dataset evidence for technician review; never changes model predictions.

Reference bands describe uploaded data, not engineering limits. Raw traces retain
original sample indices and include extreme samples even when downsampled.
"""
from __future__ import annotations
import math
from copy import deepcopy


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def metric(key, label, value, unit='source units'):
    return dict(key=key, label=label, value=number(value), unit=unit)


def compare(items):
    """Add per-metric mean ±2 population SD, within an explicit peer group."""
    from statistics import fmean, pstdev
    pools = {}
    for item in items:
        for m in item['metrics']:
            if m['value'] is not None:
                pools.setdefault((item.get('group', 'all'), m['key']), []).append(m['value'])
    for item in items:
        for m in item['metrics']:
            values = pools.get((item.get('group', 'all'), m['key']), [])
            m.update(n=len(values), mean=fmean(values) if values else None, low=None, high=None, unusual=False)
            if len(values) >= 3:
                mean, sd = fmean(values), pstdev(values)
                m.update(low=mean-2*sd, high=mean+2*sd)
                m['unusual'] = m['value'] is not None and abs(m['value']-mean) > 2*sd + 1e-12
    return items


def trace(values, label, unit='source units', times=None, sample_scope='recording'):
    import numpy as np
    a = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(a))
    if not len(valid):
        return dict(label=label, unit=unit, points=[], extremes=[], total=len(a), flagged_count=0)
    mean, sd = float(a[valid].mean()), float(a[valid].std())
    flags = valid[np.abs(a[valid]-mean) > 3*sd + 1e-12]
    extremes = sorted(flags, key=lambda i: abs(a[i]-mean), reverse=True)[:24]
    indices = sorted(set(np.linspace(0, len(a)-1, min(len(a), 180), dtype=int).tolist() + extremes + [int(valid[np.argmin(a[valid])]), int(valid[np.argmax(a[valid])])]))
    def point(i):
        return dict(index=int(i)+1, value=number(a[i]), time=str(times[i]) if times is not None else None,
                    unusual=bool(np.isfinite(a[i]) and abs(a[i]-mean) > 3*sd + 1e-12))
    return dict(label=label, unit=unit, sample_scope=sample_scope, total=len(a), mean=mean, low=mean-3*sd, high=mean+3*sd,
                flagged_count=len(flags), points=[point(i) for i in indices], extremes=[point(i) for i in extremes])


def build(key, folder, rows, names):
    import numpy as np
    items = []
    if key == 'door':
        from Door.code.pipeline import load_stream, segment, basic_features, operation_of
        from Door.code.predict import _artefact
        reference = _artefact()["ref"]
        from common.metrics import format_door_time
        segments = segment(load_stream(folder / names[0]))
        lookup = {(format_door_time(s['t'].iloc[0]), format_door_time(s['t'].iloc[-1])): s for s in segments}
        for i, row in enumerate(rows):
            s = lookup[(row['start_time'], row['end_time'])]
            f, op = basic_features(s), operation_of(s)
            f.update(reference.excess_features(s))
            metrics = [metric(k, label, f[k], unit) for k,label,unit in [
                ('mid_mean','Mid-travel current','source units'),('trav_mean','Travel current','source units'),
                ('exc_mid_mean','Mid-travel deviation from trained normal profile','IQR units'),
                ('cur_peak','Peak current','source units'),('duration','Movement duration','s'),
                ('volt_mean','Mean motor voltage','source units'),('bemf_mean','Mean back-EMF','source units'),
                ('stall_rows','Stationary samples under load','samples')]]
            items.append(dict(id=str(i), file=names[0], label=f'Movement {i+1}', group=op, operation=op,
                              start=row['start_time'], end=row['end_time'], metrics=metrics,
                              traces=[trace(s['current'], 'Motor current', times=s['dt'].tolist(), sample_scope='sequence'),
                                      trace(s['pos'], 'Door position', times=s['dt'].tolist(), sample_scope='sequence')]))
    elif key == 'acv':
        from ACV.code.pipeline import load_case, car_features, prepare
        from ACV.code.predict import _artefact
        response = _artefact()["response"]
        for row in rows:
            long,cars,_ = load_case(folder / row['file_id'])
            features = car_features(long, cars, response)
            prepared = prepare(long)
            for car in cars:
                s = prepared[prepared.car == car]
                f = features.loc[car]
                metrics = [metric(k,label,f[k],unit) for k,label,unit in [
                    ('t_in_mean','Cabin temperature while cooling','°C'),('peer_dev_mean','Temperature above peer median','°C'),
                    ('peer_dev_p90','90th percentile above peer temperature','°C'),
                    ('resid_mean','Temperature change above learned response','°C'),('set_err_mean','Temperature above cooling setpoint','°C'),('dT_mean_cool','Temperature change over 10 samples','°C'),
                    ('p_high_def','High-pressure deficit vs peers','source units'),('invalid_frac','Invalid temperature fraction','fraction')]]
                for col,label in [('p_high_1','System 1 high pressure'),('p_high_2','System 2 high pressure')]:
                    if col in s: metrics.append(metric(col,label,s[col].mean()))
                items.append(dict(id=row['file_id']+'::'+car,file=row['file_id'],car=car,label='Car '+car,
                    group=row['file_id'], start=str(s.time.min()),end=str(s.time.max()),metrics=metrics,
                    traces=[trace(s.t_in,'Valid cabin temperature','°C',s.time.tolist(), sample_scope='car time series')]))
    elif key == 'rail':
        from rail_corrugation.pipeline import load_file, speed, channel_features, channel_side
        for row in rows:
            pulse,vib,shock = load_file(folder / row['file_id'])
            v = speed(pulse); ct = channel_features(vib,shock,v)
            metrics = [metric('speed','Estimated speed',v,'m/s')]
            channels, traces = [], []
            for side in (1,2):
                sub = ct[ct.side == side]
                for k,label in [('rms','Vibration RMS'),('shock_rms','Shock RMS'),('dom_lambda','Dominant wavelength')]:
                    metrics.append(metric(f's{side}_{k}',f'Side {"I" if side==1 else "II"} · {label}',sub[k].mean(),'m' if k=='dom_lambda' else 'source units'))
                channel = int(sub.rms.idxmax())
                traces.append(trace(vib[:,channel],f'Side {"I" if side==1 else "II"} · Car {channel//8+1:02} / sensor {channel%8+1} vibration'))
            for k,c in ct.iterrows():
                channels.append(dict(id=str(k),label=f'Car {k//8+1:02} / sensor {k%8+1} / Side {"I" if channel_side(k)==1 else "II"}',
                                     metrics=[metric('rms','Vibration RMS',c.rms),metric('shock','Shock RMS',c.shock_rms)]))
            items.append(dict(id=row['file_id'],file=row['file_id'],label=row['file_id'],group='rail',
                              samples=len(pulse),metrics=metrics,traces=traces,channels=compare(channels)))
    elif key == 'shm':
        from SHM.code.pipeline import load_series, cycles
        for row in rows:
            x=load_series(folder / row['file_id']); c=cycles(x)
            metrics=[metric('damage','Estimated fatigue damage',row['prediction'],'unitless'),
                     metric('mean','Mean stress',np.mean(x)),metric('rms','Stress RMS',np.sqrt(np.mean(x*x))),
                     metric('range','Peak-to-peak stress',np.ptp(x)),metric('cycle_range','Largest stress-cycle range',c[:,0].max() if len(c) else 0),
                     metric('cycles','Counted stress cycles',c[:,2].sum() if len(c) else 0,'cycles')]
            items.append(dict(id=row['file_id'],file=row['file_id'],label=row['file_id'],group='shm',samples=len(x),
                              metrics=metrics,traces=[trace(x,'Stress')]))
    return dict(version=1,items=compare(items))


def merge(evidence):
    return dict(version=1,items=compare(deepcopy([item for e in evidence for item in e.get('items',[])])))
