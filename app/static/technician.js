'use strict';
// Technician views share the app's state and render only completed checks.
function num(v){return v===null||v===undefined||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString(undefined,{maximumSignificantDigits:5});}
function stem(name){return String(name).replace(/\.(csv|xlsx)$/i,'');}
function pill(value){return `<span class="pill ${value==='Normal'?'ok':'flag'}">${esc(value)}</span>`;}
function evidenceItem(run,id){return run.evidence?.items?.find(x=>x.id===String(id));}
function renderResults(run){
  const system=systems[run.system];state.resultSort=null;
  $('#results').hidden=false;
  $('#results').innerHTML=`<div class="results-header"><div><span class="eyebrow">COMPLETED CHECK</span><h2>${system.name} results</h2></div><div class="results-actions"><a class="button secondary" href="/api/runs/${run.id}/report">↓ JSON</a><a class="button primary" href="/api/runs/${run.id}/csv">↓ CSV</a></div></div><p class="result-meta">${esc(run.files?.length>3?run.files.length+' recordings':sourceLabel(run))} · ${esc(new Date(run.created).toLocaleString())}</p><div id="result-overview"></div><div id="result-list"></div><section id="item-detail" class="item-detail" aria-live="polite"></section>`;
  if(run.system==='door'){
    const flagged=run.rows.filter(r=>r.prediction!=='Normal').length;
    $('#result-overview').innerHTML=`<div class="chart-title"><strong>${run.rows.length} sequences · ${flagged} abnormal</strong><div class="status-legend"><span class="status-dot ok"></span>Normal <span class="status-dot bad"></span>Abnormal</div></div><div class="sequence-grid">${run.rows.map((r,i)=>`<button class="sequence-tile ${r.prediction==='Normal'?'ok':'bad'}" data-sequence="${i}" aria-pressed="false" aria-label="Sequence ${i+1}, ${esc(r.prediction)}"><small>SEQUENCE</small><strong>${String(i+1).padStart(2,'0')}</strong><span>${r.prediction==='Normal'?'Normal':'Abnormal'}</span></button>`).join('')}</div>`;
    $('#results').querySelectorAll('[data-sequence]').forEach(b=>b.onclick=()=>selectSequence(run,Number(b.dataset.sequence)));
    selectSequence(run,Math.max(0,run.rows.findIndex(r=>r.prediction!=='Normal')));
  }else if(run.system==='acv'){
    $('#result-overview').innerHTML=`<div class="chart-title"><strong>Car overview</strong><label class="filters">Recording <select id="acv-case" aria-label="Air-conditioning recording">${run.rows.map((r,i)=>`<option value="${i}">${esc(r.file_id)}</option>`).join('')}</select></label></div><div id="car-overview"></div><h3 class="compact-heading">Inspection order</h3><ol id="inspection-order" class="inspection-list"></ol><p class="data-note">Red: inspect first · Green: lower priority. Ranking does not confirm a leak.</p>`;
    $('#acv-case').onchange=()=>renderCars(run,Number($('#acv-case').value));renderCars(run,0);
  }else{
    if(run.system==='shm'){
      const sorted=[...run.rows].sort((a,b)=>Number(b.prediction)-Number(a.prediction)),max=Math.max(...sorted.map(r=>Number(r.prediction)),1e-12);
      $('#result-overview').innerHTML=`<div class="chart-title"><strong>Estimated fatigue damage</strong><span class="muted">Unitless · highest first</span></div><div class="damage-chart">${sorted.map(r=>`<button class="damage-row" data-file="${esc(r.file_id)}" title="${esc(r.file_id)}"><span class="damage-name">${esc(stem(r.file_id))}</span><span class="damage-track"><i style="width:${Math.max(0.3,100*Number(r.prediction)/max)}%"></i></span><strong>${num(r.prediction)}</strong></button>`).join('')}</div>`;
      $('#result-overview').querySelectorAll('[data-file]').forEach(b=>b.onclick=()=>inspectFile(run,b.dataset.file));
    }
    $('#result-list').innerHTML=`<div class="table-tools"><h3>Recordings</h3><div class="filters"><input id="result-search" type="search" placeholder="Find a file" aria-label="Find a file">${run.system==='rail'?'<select id="result-filter" aria-label="Rail condition filter"><option value="all">All conditions</option><option value="Normal">Normal</option><option value="Side I">Side I</option><option value="Side II">Side II</option></select>':''}</div></div><div class="table-wrap"><table><thead id="result-head"></thead><tbody id="result-body"></tbody></table></div>`;
    $('#result-search').oninput=()=>renderTable(run);if($('#result-filter'))$('#result-filter').onchange=()=>renderTable(run);
    renderTable(run);inspectFile(run,run.system==='shm'?[...run.rows].sort((a,b)=>Number(b.prediction)-Number(a.prediction))[0].file_id:run.rows[0].file_id);
  }
}
function selectSequence(run,i){
  const r=run.rows[i];$('#results').querySelectorAll('[data-sequence]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.sequence)===i)));
  showEvidence(run,evidenceItem(run,i),`Sequence ${String(i+1).padStart(2,'0')}`,r.prediction,
    `<div class="identity-times"><div><small>START</small><strong>${esc(displayTime(r.start_time))}</strong></div><div><small>END</small><strong>${esc(displayTime(r.end_time))}</strong></div></div>`);
}
function renderCars(run,index){
  const row=run.rows[index],rank=row.ranked_cars.split('|'),cars=[...rank].sort((a,b)=>Number(a)-Number(b));
  $('#car-overview').innerHTML=`<div class="car-overview">${cars.map(car=>`<button class="inspect-car ${car===rank[0]?'bad':'ok'}" data-car="${car}" aria-pressed="false"><span>CAR</span><strong>${car}</strong><small>${car===rank[0]?'Inspect first':'Lower priority'}</small><i aria-hidden="true"></i></button>`).join('')}</div>`;
  $('#inspection-order').innerHTML=rank.map(car=>`<li><button data-ranked-car="${car}">Car ${car}</button></li>`).join('');
  const select=car=>{
    $('#car-overview').querySelectorAll('[data-car]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.car===car)));
    showEvidence(run,evidenceItem(run,row.file_id+'::'+car),'Car '+car,car===rank[0]?'Inspect first':'Lower priority',`<p class="identity-file">${esc(row.file_id)} · Inspection priority ${rank.indexOf(car)+1} of ${rank.length}</p>`);
  };
  $('#car-overview').querySelectorAll('[data-car]').forEach(b=>b.onclick=()=>select(b.dataset.car));
  $('#inspection-order').querySelectorAll('button').forEach(b=>b.onclick=()=>select(b.dataset.rankedCar));select(rank[0]);
}
function renderTable(run){
  const search=$('#result-search').value.toLowerCase(),filter=$('#result-filter')?.value||'all';
  let rows=run.rows.filter(r=>r.file_id.toLowerCase().includes(search)&&(filter==='all'||r.prediction===filter));
  const sort=state.resultSort;
  if(sort)rows=[...rows].sort((a,b)=>sort.key==='file'?sort.direction*a.file_id.localeCompare(b.file_id,undefined,{numeric:true}):sort.key==='damage'?sort.direction*(Number(b.prediction)-Number(a.prediction)):sort.direction*(Number(b.prediction===sort.key)-Number(a.prediction===sort.key))||a.file_id.localeCompare(b.file_id,undefined,{numeric:true}));
  const headings=run.system==='rail'?[['file','Source file'],['Normal','Normal'],['Side I','Side I'],['Side II','Side II']]:[['file','Source file'],['damage','Estimated fatigue damage']];
  $('#result-head').innerHTML=`<tr>${headings.map(([k,label])=>`<th scope="col" aria-sort="${sort?.key===k?((sort.direction===1)===(k==='file')?'ascending':'descending'):'none'}"><button data-sort="${k}">${label} ${sort?.key===k?(sort.direction===1?'↓':'↑'):'↕'}</button></th>`).join('')}</tr>`;
  $('#result-body').innerHTML=rows.map(r=>`<tr data-file="${esc(r.file_id)}"><td><button class="file-link" data-open-file="${esc(r.file_id)}">${esc(r.file_id)}</button></td>${run.system==='rail'?['Normal','Side I','Side II'].map(label=>`<td><span class="status-cell ${r.prediction===label?(label==='Normal'?'ok':'bad'):'inactive'}">${r.prediction===label?'● ':''}${label}</span></td>`).join(''):`<td>${num(r.prediction)}</td>`}</tr>`).join('')||'<tr><td colspan="4">No matching recordings.</td></tr>';
  $('#result-head').querySelectorAll('[data-sort]').forEach(b=>b.onclick=()=>{state.resultSort={key:b.dataset.sort,direction:sort?.key===b.dataset.sort?-sort.direction:1};renderTable(run);});
  $('#result-body').querySelectorAll('[data-file]').forEach(row=>row.onclick=()=>{inspectFile(run,row.dataset.file);$('#item-detail').scrollIntoView({behavior:'smooth',block:'nearest'});});
}
function inspectFile(run,name){const row=run.rows.find(r=>r.file_id===name);showEvidence(run,evidenceItem(run,name),stem(name),run.system==='shm'?'Damage '+num(row.prediction):row.prediction,`<p class="identity-file">${esc(name)}</p>`);}
function deviation(m){
  if(m.value===null||m.value===undefined||m.mean===null||m.mean===undefined||m.n<3||m.low===null||m.high===null)return null;
  const sd=(m.high-m.low)/4;
  if(!Number.isFinite(sd)||sd<0)return null;
  if(sd===0)return m.value===m.mean?0:null;
  return 100*(m.value-m.mean)/sd;
}
function deviationCell(m){
  const pct=deviation(m);
  if(pct===null||!Number.isFinite(pct))return '<span class="sd-value unavailable" title="Not enough variation or valid data">—</span>';
  const distance=Math.abs(pct)/100;
  let color=[255,255,255],ink='#536158';
  if(distance<.5){const t=distance/.5;color=[35,98,70].map(v=>Math.round(v+(255-v)*t));ink=distance<.225?'#fff':'#244c38';}
  if(distance>1){const t=Math.min(1,distance-1);color=[157,48,48].map(v=>Math.round(255+(v-255)*t));ink=t>.6?'#fff':'#792b2b';}
  const rounded=Math.round(pct),label=(rounded>0?'+':'')+rounded.toLocaleString()+'%';
  return `<span class="sd-value" style="background:rgb(${color.join(',')});color:${ink}" title="${esc(num(pct/100))} SD from dataset mean">${label}</span>`;
}
function metricTable(metrics){return `<div class="table-wrap metric-table"><table><thead><tr><th>Measurement</th><th>Selected</th><th>Dataset mean</th><th title="100 × (selected − mean) ÷ standard deviation">Deviation (% of SD)</th><th>Dataset band (±2 SD)</th></tr></thead><tbody>${metrics.map(m=>`<tr><td>${esc(m.label)}<small>${esc(m.unit)}</small></td><td title="${esc(m.value)}">${num(m.value)}</td><td>${num(m.mean)} <small>n=${m.n||0}</small></td><td>${deviationCell(m)}</td><td>${m.low===null?'—':`${num(m.low)} – ${num(m.high)}`}</td></tr>`).join('')}</tbody></table></div>`;}
function showEvidence(run,item,title,status,identity){
  const bad=status==='Abnormal resistance'||status==='Inspect first'||status==='Side I'||status==='Side II';
  const body=item?`${metricTable(item.metrics)}${item.traces.map((t,i)=>renderTrace(t,i,run.system)).join('')}${item.channels?`<details class="raw-details"><summary>All 64 sensor positions</summary><div class="table-wrap"><table><thead><tr><th>Position</th><th>Vibration RMS</th><th>Shock RMS</th><th>Dataset comparison</th></tr></thead><tbody>${item.channels.map(c=>`<tr><td>${esc(c.label)}</td>${c.metrics.map(m=>`<td>${num(m.value)}</td>`).join('')}<td>${c.metrics.some(m=>m.unusual)?'Outside ±2 SD':'Within ±2 SD'}</td></tr>`).join('')}</tbody></table></div></details>`:''}`:`<p class="muted">${esc(run.evidence?.error||'Recheck the original recording to add measurements to this saved result.')}</p>`;
  $('#item-detail').innerHTML=`<div class="detail-heading"><h3>${esc(title)}</h3><span class="detail-status ${bad?'bad':status==='Normal'||status==='Lower priority'?'ok':'neutral'}">${esc(status)}</span></div>${identity}${item?.start&&run.system!=='door'?`<p class="recording-period">${esc(displayTime(item.start))} → ${esc(displayTime(item.end))}</p>`:''}${item?.samples?`<p class="data-note">${item.samples.toLocaleString()} original samples</p>`:''}${body}`;
}
function renderTrace(t,index,system){
  const scope=t.sample_scope||(system==='door'?'sequence':system==='acv'?'car time series':'recording');
  if(!t.points?.some(p=>p.value!==null))return `<p class="data-note">${esc(t.label)}: no valid readings.</p>`;
  const values=t.points.filter(p=>p.value!==null).map(p=>p.value),lo=Math.min(...values,t.low),hi=Math.max(...values,t.high),span=hi-lo||1;
  const x=i=>60+(i-1)/Math.max(1,t.total-1)*680,y=v=>165-(v-lo)/span*140;
  let path='',pen=false;
  for(const p of t.points){if(p.value===null){pen=false;continue;}path+=`${pen?'L':'M'}${x(p.index).toFixed(2)},${y(p.value).toFixed(2)} `;pen=true;}
  return `<details class="raw-details"><summary>${esc(t.label)} · ${t.flagged_count} readings outside ±3 SD</summary><div class="trace-wrap"><svg class="raw-trace" viewBox="0 0 780 205" role="img" aria-label="${esc(t.label)}, sample index within ${esc(scope)}, ${esc(t.unit)}"><rect x="60" y="${y(t.high)}" width="680" height="${y(t.low)-y(t.high)}" fill="#edf3ed"/><line x1="60" x2="740" y1="${y(t.mean)}" y2="${y(t.mean)}" stroke="#91a797" stroke-dasharray="4 4"/><path d="${path}" fill="none" stroke="#3d7461" stroke-width="1.5"/>${t.points.filter(p=>p.unusual&&p.value!==null).map(p=>`<circle cx="${x(p.index)}" cy="${y(p.value)}" r="3" fill="#b54848"><title>Sample ${p.index}: ${p.value}</title></circle>`).join('')}<text x="5" y="28">${num(hi)}</text><text x="5" y="168">${num(lo)}</text><text x="60" y="191">1</text><text x="640" y="191">${t.total} samples</text></svg></div><p class="data-note">Samples within ${esc(scope)} · ${esc(t.unit)} · Mean ${num(t.mean)} · ±3 SD ${num(t.low)} to ${num(t.high)} · ${t.points.length} selected original points shown</p>${t.extremes.length?`<div class="table-wrap"><table><thead><tr><th>Sample</th>${t.extremes[0].time?'<th>Recording time</th>':''}<th>Original value</th></tr></thead><tbody>${t.extremes.map(p=>`<tr><td>${p.index}</td>${p.time?`<td>${esc(displayTime(p.time))}</td>`:''}<td>${esc(p.value)}</td></tr>`).join('')}</tbody></table></div><p class="data-note">Largest deviations first · up to 24 readings</p>`:'<p class="data-note">No readings outside this trace’s band.</p>'}</details>`;
}
let exportChoices=[];
function openExport(){
  exportChoices=[];const seen=new Set();
  for(const r of historyRuns()){
    if(r.system==='door'){if(seen.has('door'))continue;seen.add('door');exportChoices.push({id:r.id,system:r.system,label:sourceLabel(r)});}
    else for(const row of r.rows){const key=r.system+':'+row.file_id;if(seen.has(key))continue;seen.add(key);exportChoices.push({id:r.id,system:r.system,file:row.file_id,label:row.file_id});}
  }
  $('#export-checklist').innerHTML=Object.entries(systems).map(([key,s])=>{const count=exportChoices.filter(c=>c.system===key).length;return `<label class="task-export"><input type="checkbox" data-task="${key}" ${count?'checked':'disabled'}><span>${s.name}</span><small>${count?`${count} recording${count===1?'':'s'}`:'No results'}</small></label>`;}).join('');
  const update=()=>{$('#download-zip').disabled=!$('#export-checklist').querySelectorAll('input:checked').length;};
  $('#export-checklist').querySelectorAll('input').forEach(c=>c.onchange=update);update();$('#export-error').hidden=true;$('#export-dialog').showModal();
}
async function downloadZip(){
  const button=$('#download-zip');button.disabled=true;
  const tasks=new Set([...$('#export-checklist').querySelectorAll('input:checked')].map(c=>c.dataset.task));
  const selection=exportChoices.filter(c=>tasks.has(c.system)).map(({id,file})=>({id,file}));
  try{
    const response=await fetch('/api/selected-results',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({selection})});
    if(!response.ok){await responseJSON(response);throw new Error('Download failed.');}
    if(!response.headers.get('Content-Type')?.includes('application/zip'))throw new Error('The server did not return a results ZIP.');
    const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');a.href=url;a.download='selected_check_results.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('#export-dialog').close();toast('Selected results downloaded.');
  }catch(error){$('#export-error').textContent=error.message;$('#export-error').hidden=false;}finally{button.disabled=!selection.length;}
}
