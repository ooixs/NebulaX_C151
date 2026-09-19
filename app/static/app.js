'use strict';
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icons = {
 door:'<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M12 3v18M9 11v3m6-3v3M2 21h20"/>',
 acv:'<path d="M12 2v20M3.3 7l17.4 10M3.3 17L20.7 7M9 4l3 3 3-3M9 20l3-3 3 3M4 10l4-1-1-4M20 14l-4 1 1 4M4 14l4 1-1 4M20 10l-4-1 1-4"/>',
 rail:'<path d="M7 3L4 22M17 3l3 19M6 7h12M5 12h14M4 17h16M3 22h18"/>',
 shm:'<path d="M3 18V6m18 12V6M3 12h4l3-7 4 14 3-7h4M3 22h18"/>'
};
const systems = {
 door:{name:'Doors',short:'Door',subtitle:'Find faulty doors',extension:'.csv',title:'Find door movements that need inspection.',description:'The app finds each opening or closing movement and checks for unusual resistance that could indicate a faulty door. This check detects resistance faults, not every type of door fault.',output:'Start and end times, with a result for each movement',focus:'Movements marked “Abnormal resistance”',metric:'IoU-weighted F1 · checks movement timing and fault labels',action:'Review flagged movements, then check the door tracks, seals and motor.',hint:'Add one CSV file containing a continuous door recording.',format:'Use one CSV file with the original column headings and all 17 columns, including time, motor readings and door position.',method:'Finds individual door movements and compares their motor readings with learned patterns.',limitation:'The app may miss a movement or mark its start or end incorrectly. “Normal” means no unusual resistance was detected; it does not confirm the door is fault-free.'},
 acv:{name:'Air conditioning',short:'ACV',subtitle:'Find the car most likely to have a refrigerant leak',extension:'.xlsx',title:'See which car to inspect first.',description:'The app compares air-conditioning readings across the train and lists the cars from most to least likely to have a refrigerant leak.',output:'An ordered list of cars for each recording',focus:'The first car in each list',metric:'Rank-decay score · rewards placing the faulty car near the top',action:'Start with the first car in the list, then check the others in order.',hint:'Add Excel files (.xlsx). Each file must include readings for every car.',format:'Keep the original Excel column headings, such as “Car 03 - ACV Running Mode”, and the readings for every car. Different files may contain different measurements.',method:'Compares each car’s temperature with the other cars and uses pressure readings when available.',limitation:'The model was developed using six known fault cases. The order helps you decide where to inspect first; it is not a confirmed leak diagnosis or a percentage chance of a leak.'},
 rail:{name:'Rail condition',short:'Rail',subtitle:'Find rail corrugation on either side',extension:'.csv',title:'Find which rail side needs inspection.',description:'The app checks vibration readings for corrugation: a repeated pattern of uneven rail wear. Results show Normal, Side I or Side II.',output:'A corrugation result for each recording',focus:'Results marked Side I or Side II',metric:'Macro F1 · gives equal weight to each of the three results',action:'Match flagged files to their track locations, then inspect the indicated rail side.',hint:'Add one or more CSV files, each with a one-second vibration recording.',format:'Use CSV files with the original headings and all 129 columns: one speed signal plus 128 vibration and shock measurements. Each recording covers one second at 10,000 readings per second.',method:'Adjusts vibration measurements for speed, then checks each rail side for learned wear patterns.',limitation:'The model was trained with relatively few faulty examples. A Normal result does not rule out every rail defect. Use your recording records to find the track location.'},
 shm:{name:'Structural health',short:'SHM',subtitle:'Estimate damage from repeated stress',extension:'.csv',title:'Compare damage caused by repeated stress.',description:'The app estimates the fatigue damage accumulated during each equal-length stress recording. The supplied data covers healthy operation on two lines under load conditions AW0 and AW4.',output:'One fatigue damage estimate for each file',focus:'Damage per recording, not a structural fault diagnosis',metric:'1 − MAPE, minimum 0 · higher scores mean lower percentage error',action:'Before comparing values, confirm the same measurement point, line and load condition (AW0 or AW4). File numbers are random; they do not show recording order.',hint:'Add stress recordings as CSV files without column headings.',format:'Use CSV files with no heading row. The first column must contain numeric stress readings. The app returns one damage estimate per file.',method:'Counts stress cycles and estimates fatigue damage, with a correction learned from training examples.',limitation:'The dataset contains healthy operation, not examples of structural failure. A larger estimate is not a confirmed fault or a remaining-life forecast. Line, load and recording order cannot be recovered from filenames.'}
};
const state = {system:'rail',files:[],models:{},runs:[],current:null,busy:false,view:'workspace',bySystem:{},filesBySystem:{},detailId:null,progress:''};
let toastTimer;
function toast(message){$('#toast').textContent=message;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,4200);}
function showError(message){$('#error').textContent=message;$('#error').hidden=!message;}
async function responseJSON(response){
  const text=await response.text(); let data;
  try { data=JSON.parse(text); } catch {
    if(response.status===413) throw new Error('The upload is too large for this server. Try one smaller file.');
    if(response.status===501||response.status===405) throw new Error('This server cannot check uploaded files. Start the app using app/server.py, then open its local address.');
    throw new Error(`The server did not return a check result${response.status?' (HTTP '+response.status+')':''}. Try again with one file. Any completed checks are in Previous results.`);
  }
  if(!response.ok) throw new Error(data.error||'The check could not be completed. Try again.');
  return data;
}
async function api(url,options){return responseJSON(await fetch(url,options));}
function setView(view){
  state.view=view;window.scrollTo({top:0,behavior:'instant'});
  document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!==`${view}-view`);
  document.querySelectorAll('.nav').forEach(el=>el.classList.toggle('active',el.dataset.view===(view==='detail'?'history':view)));
  $('#page-name').textContent={workspace:'Check train condition',history:'Previous results',detail:'Previous results / Check details',guide:'How to use this app'}[view];
  if(view==='history')renderHistory();
  if(view==='workspace'){
    $('#workspace-results-host').append($('#results'));
    state.current=state.bySystem[state.system]||null;
    if(state.current)renderResults(state.current);else renderEmpty();
  }
}
function selectSystem(key){
  if(state.busy)return;
  state.filesBySystem[state.system]=state.files;
  state.system=key;state.files=state.filesBySystem[key]||[];$('#file-input').value='';
  showError('');renderWorkspace();
  state.current=state.bySystem[key]||null;
  if(state.current)renderResults(state.current);else renderEmpty();
}
function renderWorkspace(){const s=systems[state.system];$('#system-cards').innerHTML=Object.entries(systems).map(([key,item])=>`<button class="system-card ${key===state.system?'selected':''}" data-system="${key}" aria-pressed="${key===state.system}" ${state.busy?'disabled':''}><span class="system-icon"><svg viewBox="0 0 24 24" aria-hidden="true">${icons[key]}</svg></span><span><strong>${item.name}</strong><small>${item.subtitle}</small></span>${key===state.system?'<i class="selected-dot"></i>':''}</button>`).join('');$('#system-cards').querySelectorAll('button').forEach(b=>b.onclick=()=>selectSystem(b.dataset.system));$('#input-type').textContent=s.extension.slice(1).toUpperCase();$('#input-hint').textContent=s.hint;$('#file-input').accept=s.extension;$('#file-input').multiple=state.system!=='door';$('#file-input').disabled=state.busy;$('#upload-limit').textContent=`${s.extension.slice(1).toUpperCase()} · ${state.system==='door'?'one continuous recording':'up to 100 files'} · 120 MB per file`;$('#context-title').textContent=s.title;$('#context-description').textContent=s.description;$('#asset-tag').textContent=s.short.toUpperCase();$('#output-summary').textContent=s.output;$('#metric-summary').textContent=s.focus;$('#action-summary').textContent=s.action;$('#required-columns').innerHTML=columnGuide(state.system);renderFiles();renderNotice();}
function renderNotice(){
  const model=state.models[state.system],notice=$('#model-notice');
  notice.hidden=!model||model.ready;
  notice.textContent='This check is unavailable. Ask the app maintainer to restore its model, then refresh this page.';
  $('#analyze').disabled=state.busy||!state.files.length||!model?.ready;
  $('#analyze').innerHTML=state.busy?'Checking…':state.files.length>1?'Check files <span>↗</span>':'Check file <span>↗</span>';
  $('#progress').hidden=!state.busy;$('#progress-text').textContent=state.progress;
}
async function refreshStatus(){try{state.models=(await api('/api/status')).models;renderNotice();}catch(error){$('#model-notice').textContent='The app has lost its connection. Restart the app and refresh this page.';showError(error.message);}}
function renderFiles(){$('#dropzone').hidden=state.system==='door'&&state.files.length>0;$('#file-list').innerHTML=state.files.map((file,i)=>`<div class="file-item"><span title="${esc(file.name)}">▤ ${esc(file.name)}</span><small>${file.size<1048576?(file.size/1024).toFixed(0)+' KB':(file.size/1048576).toFixed(1)+' MB'}</small><button class="icon-button" aria-label="Remove ${esc(file.name)}" data-index="${i}" ${state.busy?'disabled':''}>×</button></div>`).join('');$('#file-list').querySelectorAll('button').forEach(b=>b.onclick=()=>{state.files.splice(Number(b.dataset.index),1);renderFiles();renderNotice();});}
function addFiles(files){
  if(state.busy)return;showError('');const next=[...state.files,...files],s=systems[state.system];
  if(next.some(f=>!f.name.endsWith(s.extension))){showError(`Choose ${s.extension} files for ${s.name}. Keep the original column layout.`);return;}
  if(next.some(f=>!f.size)){showError('One file is empty. Remove it and choose a recording containing sensor readings.');return;}
  if(new Set(next.map(f=>f.name.toLowerCase())).size!==next.length){showError('A filename is repeated. Remove the duplicate before continuing.');return;}
  if(next.length>(state.system==='door'?1:100)){showError(state.system==='door'?'For Doors, check one continuous recording at a time. Remove the selected file to replace it.':'Choose up to 100 files at a time.');return;}
  if(next.some(f=>f.size>120*1048576)){showError('One file exceeds 120 MB. Choose a smaller recording.');return;}
  state.files=next;state.filesBySystem[state.system]=next;renderFiles();renderNotice();
}
function addRun(run){
  state.runs=state.runs.filter(r=>r.id!==run.id);state.runs.unshift(run);
  state.bySystem[run.system]=run;state.current=run;updateCounts();
  if(state.view==='workspace'&&state.system===run.system)renderResults(run);
}
function updateCounts(){ $('#history-count').textContent=historyRuns().length;$('#export-count').textContent=`${Object.keys(latestLive()).length} / 4`; }
function historyRuns(){const merged=new Set(state.runs.flatMap(r=>r.combined_from||[]));return state.runs.filter(r=>!r.preview&&!merged.has(r.id));}
function renderEmpty(){
  $('#results').hidden=false;$('#results').innerHTML='<div class="empty-result"><span aria-hidden="true">▤</span><h2>Your results show up here</h2><p>Add a recording above, then select Check file. You can switch systems without losing completed results.</p></div>';
}
function latestLive(){const latest={};for(const run of state.runs){if(run.preview)continue;if(!latest[run.system])latest[run.system]={rows:[],names:new Set()};const group=latest[run.system];if(run.system==='door'){if(!group.rows.length)group.rows=run.rows;}else for(const row of run.rows){if(!group.names.has(row.file_id)){group.names.add(row.file_id);group.rows.push(row);}}}return latest;}
async function analyze(){
  if(state.busy)return;const key=state.system,files=[...state.files],completed=[];
  state.busy=true;showError('');state.progress='Preparing your files…';renderWorkspace();
  try {
    for(let i=0;i<files.length;i++){
      state.progress=`Checking file ${i+1} of ${files.length}: ${files[i].name}`;renderNotice();
      const data=new FormData();data.append('system',key);data.append('files',files[i],files[i].name);
      const run=await api('/api/analyze',{method:'POST',body:data});completed.push(run);addRun(run);
    }
    if(completed.length>1){
      state.progress='Bringing your results together…';renderNotice();
      const run=await api('/api/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:completed.map(r=>r.id)})});addRun(run);
    }
    state.files=[];state.filesBySystem[key]=[];toast('Check complete. Your results have been saved.');
    if(state.view==='workspace')$('#results').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(error){
    const done=new Set(completed.flatMap(r=>r.files.map(f=>f.name)));
    state.files=files.filter(f=>!done.has(f.name));state.filesBySystem[key]=state.files;
    showError(`${error.message}${completed.length?' Completed files are saved in Previous results. Only unfinished files remain selected.':''}`);
  }finally{state.busy=false;renderWorkspace();refreshStatus();if(state.view==='history')renderHistory();}
}
function preview(){setView('history');}
function summary(run){const rows=run.rows,key=run.system;if(key==='rail'||key==='door'){const abnormal=rows.filter(r=>r.prediction!=='Normal').length;return [{label:key==='door'?'Door movements found':'Recordings checked',value:rows.length,detail:key==='door'?'In this recording':'Files in this check'},{label:'Needs inspection',value:abnormal,detail:key==='door'?'Abnormal resistance':'Uneven wear on one rail side'},{label:'Marked Normal',value:rows.length-abnormal,detail:'No target fault detected'}];}if(key==='acv'){const first=rows[0].ranked_cars.split('|');return [{label:'Recordings checked',value:rows.length,detail:'One car list per recording'},{label:'First recording · inspect first',value:`Car ${first[0]}`,detail:'Most likely to have a refrigerant leak'},{label:'Cars in first recording',value:first.length,detail:'Listed in inspection order'}];}const vals=rows.map(r=>+r.prediction);return [{label:'Recordings checked',value:rows.length,detail:'Files containing stress readings'},{label:'Highest damage estimate',value:Math.max(...vals).toFixed(4),detail:'A damage value with no units'},{label:'Middle damage estimate',value:median(vals).toFixed(4),detail:'Median across these files'}];}
function median(values){const a=[...values].sort((a,b)=>a-b);return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;}
function insight(run){const rows=run.rows;if(run.system==='rail'){const n=rows.filter(r=>r.prediction!=='Normal').length;const side1=rows.filter(r=>r.prediction==='Side I').length;return n?`${n} recording${n===1?' needs':'s need'} review: ${side1} on Side I and ${n-side1} on Side II. Match each recording to its track location, then inspect the indicated side for corrugation.`:'No rail corrugation was detected in these recordings. Continue routine inspections; these results do not rule out all rail defects.';}if(run.system==='door'){const n=rows.filter(r=>r.prediction!=='Normal').length;return n?`${n} door movements show unusual resistance. Use their start and end times to find them in the recording, then inspect the tracks, seals and motor.`:'No unusual resistance was detected. Check that the app found every opening and closing movement in the original recording.';}if(run.system==='acv')return 'For each recording, inspect the first car in the list, then work through the others. Check cooling performance and the refrigerant circuit. The list shows inspection order, not the probability of a leak.';const high=rows.reduce((a,b)=>+a.prediction>+b.prediction?a:b);return `${high.file_id} has the highest estimated damage (${(+high.prediction).toFixed(4)}). The supplied segments are equal in length and were recorded during healthy operation. Before comparing values, verify the measurement point, line and AW0/AW4 load condition. This is not a fault diagnosis.`;}
function chart(run){const rows=run.rows;if(run.system==='rail'){const colors=['#91a87c','#ce8d66','#496d5b'];const labels=['Normal','Side I','Side II'];return `<div class="chart-title"><strong>Results by rail condition</strong><div class="legend"><span><i></i>Normal</span><span><i class="orange"></i>Side I</span><span><i class="dark"></i>Side II</span></div></div><div class="distribution" role="img" aria-label="${labels.map(l=>`${l}: ${rows.filter(r=>r.prediction===l).length}`).join(', ')}">${labels.map((l,i)=>{const n=rows.filter(r=>r.prediction===l).length;return n?`<div style="width:${n/rows.length*100}%;background:${colors[i]}" title="${l}: ${n}">${n}</div>`:'';}).join('')}</div><p class="data-note">Each file has one result. Side I uses sensors at wheel positions 1, 3, 5 and 7; Side II uses positions 2, 4, 6 and 8.</p>`;}if(run.system==='door')return `<div class="chart-title"><strong>Door movements in recording order</strong><div class="legend"><span><i></i>Normal</span><span><i class="orange"></i>Abnormal resistance</span></div></div><div class="segments">${rows.map((r,i)=>`<button class="segment ${r.prediction!=='Normal'?'flag':''}" data-cycle="${i}" aria-label="Movement ${i+1}: ${esc(r.prediction)}; ${esc(displayTime(r.start_time))} to ${esc(displayTime(r.end_time))}" title="${esc(displayTime(r.start_time))} → ${esc(displayTime(r.end_time))}">${String(i+1).padStart(2,'0')}</button>`).join('')}</div><p id="cycle-detail" class="data-note">Select a numbered movement to see when it started and ended. Times use the recording’s clock, not a converted timezone. The boxes show order, not duration.</p>`;if(run.system==='acv')return `<div class="chart-title"><strong>Car inspection order</strong><label class="filters">Recording <select id="acv-case" aria-label="Choose an air-conditioning recording">${rows.map((r,i)=>`<option value="${i}">${esc(r.file_id)}</option>`).join('')}</select></label></div><div id="ranking" class="ranking">${ranking(rows[0])}</div><p class="data-note">Start with the car on the left: it is most likely to have a refrigerant leak. Car numbers match your original file.</p>`;const sorted=[...rows].sort((a,b)=>+b.prediction-+a.prediction).slice(0,24);const max=Math.max(...sorted.map(r=>+r.prediction),.000001);return `<div class="chart-title"><strong>Estimated fatigue damage · highest first</strong><span class="muted">Damage value · no units</span></div><div class="bars" role="img" aria-label="Highest damage estimate recordings; exact values are available in the table">${sorted.map((r,i)=>`<div class="bar-column" style="height:${(+r.prediction/max)*100}%" title="${esc(r.file_id)}: ${(+r.prediction).toPrecision(6)}"><span>${i+1}</span></div>`).join('')}</div><p class="data-note">${sorted.length} highest estimates shown. Bars are ordered by estimated damage, not recording time. They do not show how soon a part might fail.</p>`;}
function ranking(row){return row.ranked_cars.split('|').map((c,i)=>`<div class="rank-car"><small>RANK ${i+1}</small><strong>${esc(c)}</strong><span>${i===0?'INSPECT FIRST':'CAR'}</span></div>`).join('');}
function renderResults(run){const s=systems[run.system];$('#results').hidden=false;$('#results').innerHTML=`<div class="results-header"><div><span class="result-badge ${run.preview?'':'live'}">${run.preview?'EXAMPLE RESULTS · YOUR FILES WERE NOT CHECKED':'RESULTS FROM YOUR UPLOADED FILES'}</span><h2>${s.name} results</h2></div><div class="results-actions"><a class="button secondary" href="/api/runs/${run.id}/report">↓ Download JSON</a><a class="button primary" href="/api/runs/${run.id}/csv">↓ ${run.preview?'Example results (CSV)':'Download results (CSV)'}</a></div></div><p class="result-meta">${run.preview?'Previously saved predictions. The original files and model version have not been verified.':`${esc(sourceLabel(run))} · Checked ${esc(new Date(run.created).toLocaleString())}`}</p><div class="stat-grid">${summary(run).map(s=>`<div class="stat"><small>${s.label}</small><strong>${esc(s.value)}</strong><span>${s.detail}</span></div>`).join('')}</div><div class="insight"><span>↗</span><div><strong>What to check next</strong><p>${esc(insight(run))}</p></div></div>${chart(run)}<div class="table-tools"><h3>Results for each item <span class="muted">/ ${run.rows.length}</span></h3><div class="filters"><input id="result-search" type="search" placeholder="Search results…" aria-label="Search results">${['rail','door'].includes(run.system)?'<select id="result-filter" aria-label="Filter results"><option value="all">All results</option><option value="flagged">Needs inspection</option><option value="normal">Marked Normal</option></select>':''}</div></div><div class="table-wrap"><table><thead id="result-head"></thead><tbody id="result-body"></tbody></table></div><p id="row-count" class="data-note"></p><p class="data-note">${esc(s.limitation)}</p>`;$('#result-search').oninput=()=>renderTable(run);if($('#result-filter'))$('#result-filter').onchange=()=>renderTable(run);if($('#acv-case'))$('#acv-case').onchange=e=>$('#ranking').innerHTML=ranking(run.rows[Number(e.target.value)]);$('#results').querySelectorAll('[data-cycle]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.cycle),r=run.rows[i];$('#cycle-detail').textContent=`Movement ${i+1} · ${r.prediction} · ${displayTime(r.start_time)} → ${displayTime(r.end_time)} (recording time)`;});renderTable(run);}
function renderTable(run){const search=$('#result-search').value.toLowerCase(),filter=$('#result-filter')?.value||'all';const rows=run.rows.map((r,i)=>({...r,index:i+1})).filter(r=>Object.values(r).some(v=>String(v).toLowerCase().includes(search))&&(filter==='all'||(filter==='flagged'?r.prediction!=='Normal':r.prediction==='Normal')));const headers=run.system==='door'?['Movement','Start time','End time','Result']:run.system==='acv'?['Source file','Cars · inspect from left to right']:['Source file',run.system==='shm'?'Estimated fatigue damage':'Result'];$('#result-head').innerHTML=`<tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr>`;$('#result-body').innerHTML=rows.slice(0,500).map(r=>{let cells;if(run.system==='door')cells=[String(r.index).padStart(2,'0'),esc(displayTime(r.start_time)),esc(displayTime(r.end_time)),pill(r.prediction)];else if(run.system==='acv')cells=[esc(r.file_id),esc(r.ranked_cars.split('|').join(' → '))];else cells=[esc(r.file_id),run.system==='shm'?(+r.prediction).toPrecision(7):pill(r.prediction)];return `<tr>${cells.map(c=>`<td>${c}</td>`).join('')}</tr>`;}).join('')||`<tr><td colspan="${headers.length}">No results match your search or filter.</td></tr>`;$('#row-count').textContent=`Showing ${Math.min(rows.length,500)} of ${rows.length} matching results. Downloaded files include all results, with the full numeric precision returned by the model.`;}
function pill(value){return `<span class="pill ${value!=='Normal'?'flag':''}">${esc(value)}</span>`;}
function renderHistory(){
  const system=$('#history-system').value,period=$('#history-period').value;
  $('#custom-hours-label').hidden=period!=='custom';
  const hours=period==='custom'?Number($('#history-hours').value):Number(period);
  if(period==='custom'&&(!Number.isFinite(hours)||hours<=0)){$('#history-content').innerHTML='<p class="muted">Enter a number of hours greater than zero.</p>';return;}
  const cutoff=period==='all'?-Infinity:Date.now()-hours*3600000;
  const rows=historyRuns().filter(r=>(system==='all'||r.system===system)&&Date.parse(r.created)>=cutoff);
  $('#history-content').innerHTML=rows.length?rows.map(r=>`<button class="history-item history-link" data-run="${r.id}"><span class="system-icon"><svg viewBox="0 0 24 24" aria-hidden="true">${icons[r.system]}</svg></span><span class="history-description"><strong>${systems[r.system].name}</strong><span>${esc(sourceLabel(r))}</span><small>${esc(new Date(r.created).toLocaleString())}</small></span><span aria-hidden="true">View results →</span></button>`).join(''):'<div class="panel empty"><h2>No saved checks match these filters.</h2><p>Choose another time range or check a recording to add results.</p></div>';
  $('#history-content').querySelectorAll('[data-run]').forEach(b=>b.onclick=()=>openHistory(b.dataset.run));
}
function sourceLabel(run){return run.files?.length?run.files.map(f=>f.name).join(', '):'Saved recording';}
function openHistory(id){
  const run=state.runs.find(r=>r.id===id);if(!run){toast('These results are unavailable.');return;}
  state.detailId=id;setView('detail');$('#detail-results-host').append($('#results'));renderResults(run);
}

function openExport(){const latest=latestLive();$('#export-checklist').innerHTML=Object.entries(systems).map(([key,s])=>`<div class="check-row"><span>${latest[key]?'✓':'○'}</span><strong>${s.name}</strong><small>${latest[key]?`${latest[key].rows.length} results from your uploaded files`:'No uploaded files checked yet'}</small></div>`).join('');$('#download-zip').disabled=!Object.keys(latest).length;$('#download-submission').disabled=!Object.keys(latest).length;$('#export-error').hidden=true;$('#export-dialog').showModal();}
async function downloadZip(judging=false){
  const button=judging?$('#download-submission'):$('#download-zip');button.disabled=true;
  try {
    const response=await fetch(judging?'/api/export':'/api/all-results',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:state.runs.filter(r=>!r.preview).map(r=>r.id)})});
    if(!response.ok){await responseJSON(response);throw new Error('The download could not be created.');}
    if(!response.headers.get('Content-Type')?.includes('application/zip'))throw new Error('The server did not return a ZIP file. Try again.');
    const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');a.href=url;a.download=judging?'predictions.zip':'all_check_results.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('#export-dialog').close();toast('Your results have been downloaded.');
  }catch(error){$('#export-error').textContent=error.message;$('#export-error').hidden=false;}finally{button.disabled=false;}
}
function displayTime(value){
  const parts=String(value).split('-');
  if(parts.length===7&&parts.every(p=>/^\d+$/.test(p))){
    const [year,month,day,hour,minute,second,ms]=parts.map(Number),months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return `${String(day).padStart(2,'0')} ${months[month-1]} ${year}, ${[hour,minute,second].map(n=>String(n).padStart(2,'0')).join(':')}.${String(ms).padStart(3,'0')}`;
  }
  const match=String(value).match(/^(\d{4})-(\d{2})-(\d{2})[T ](.*)$/);
  return match?`${match[3]} ${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][Number(match[2])-1]} ${match[1]}, ${match[4]}`:String(value);
}
function columnGuide(key){
  const guides={
    door:'<p><b>Required:</b> all 17 columns, with a heading row, in the supplied order.</p><ol><li>Timestamp</li><li>Motor current, voltage and back-EMF</li><li>Opening and closing time</li><li>Close and open commands</li><li>DCSR, DCSL, DLSR and DLSL status flags</li><li>Opened, locked, opening and closing flags</li><li>Door position</li></ol><p>Keep the original columns even if a flag is always zero. None can be removed: the reader identifies them by position.</p>',
    rail:'<p><b>Required:</b> all 129 numeric columns, with a heading row, in the supplied order.</p><ul><li>Column 1: wheel-speed pulse signal.</li><li>Columns 2–129: vibration and shock pairs for 8 wheel positions on each of 8 cars.</li></ul><p>Order: Car 1 / Position 1 vibration, shock; Position 2 vibration, shock; continuing through Car 8 / Position 8. Each file has 10,000 readings over one second. No columns are optional.</p>',
    acv:'<p><b>Required:</b> a Time column and indoor/cabin temperature readings for every car, using the original <code>Car NN - parameter</code> headings. Use all cars from the supplied file; peer comparison needs at least three cars with valid readings.</p><p><b>Optional to the reader:</b> outdoor temperature, cooling/heating setpoints, running/setting mode, load-halved and information-valid flags, and pressure readings. These improve context when provided; keep them in original files. If a validity flag is included, also retain indoor, outdoor and cooling-setpoint temperature columns.</p><p>Do not rename car numbers or remove columns from the supplied Excel files. Two-digit identifiers, such as 03, must stay unchanged.</p>',
    shm:'<p><b>Required:</b> numeric stress readings in the first column, with no heading row.</p><p>Each supplied file is one equal-length segment from a measurement point. Additional columns are ignored by this model. Filenames do not identify the line, load condition or recording order.</p>'
  };return guides[key];
}
document.querySelectorAll('.nav').forEach(button=>button.onclick=()=>setView(button.dataset.view));$('#file-input').onchange=e=>{addFiles([...e.target.files]);e.target.value='';};const dropzone=$('#dropzone');['dragenter','dragover'].forEach(name=>dropzone.addEventListener(name,e=>{e.preventDefault();dropzone.classList.add('dragover');}));['dragleave','drop'].forEach(name=>dropzone.addEventListener(name,e=>{e.preventDefault();dropzone.classList.remove('dragover');}));dropzone.addEventListener('drop',e=>addFiles([...e.dataTransfer.files]));$('#analyze').onclick=analyze;$('#preview').onclick=preview;$('#export-open').onclick=openExport;$('#close-export').onclick=()=>$('#export-dialog').close();$('#download-zip').onclick=()=>downloadZip(false);$('#download-submission').onclick=()=>downloadZip(true);$('.train-cars').innerHTML=Array.from({length:8},(_,i)=>`<div class="car"><i></i><i></i><i></i><b>${String(i+1).padStart(2,'0')}</b></div>`).join('');$('#format-guide').innerHTML=Object.values(systems).map(s=>`<h3>${s.name}</h3><p>${s.format}</p>`).join('');renderWorkspace();refreshStatus();

async function restoreHistory(){
  try {
    state.runs=(await api('/api/history')).runs;
    for(const run of state.runs)if(!run.preview&&!state.bySystem[run.system])state.bySystem[run.system]=run;
    updateCounts();if(state.view==='history')renderHistory();
    if(state.view==='workspace'&&state.bySystem[state.system])renderResults(state.bySystem[state.system]);
  }catch(error){toast('Saved results could not be loaded. Refresh the page to try again.');}
}
$('#back-history').onclick=()=>setView('history');
$('#history-system').onchange=renderHistory;$('#history-period').onchange=renderHistory;$('#history-hours').oninput=renderHistory;
renderEmpty();restoreHistory();
