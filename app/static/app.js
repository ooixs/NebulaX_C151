'use strict';

const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? '').replace(
  /[&<>"']/g,
  (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]),
);

const icons = {
  door: '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M12 3v18M9 11v3m6-3v3M2 21h20"/>',
  acv: '<path d="M12 2v20M3.3 7l17.4 10M3.3 17L20.7 7M9 4l3 3 3-3M9 20l3-3 3 3M4 10l4-1-1-4M20 14l-4 1 1 4M4 14l4 1-1 4M20 10l-4-1 1-4"/>',
  rail: '<path d="M7 3L4 22M17 3l3 19M6 7h12M5 12h14M4 17h16M3 22h18"/>',
  shm: '<path d="M3 18V6m18 12V6M3 12h4l3-7 4 14 3-7h4M3 22h18"/>',
};

const systems = {
  door: {
    name: 'Doors', short: 'Door', extension: '.csv',
    subtitle: 'Find sticking movements',
    title: 'Find door movements that need inspection.',
    description: 'Each opening and closing movement is checked for unusual resistance.',
    output: 'Movement time and condition',
    action: 'Inspect the track, seals, and motor for flagged movements.',
    hint: 'Add one continuous door recording.',
    format: 'One CSV with the original 17 columns, including time, motor readings, and position.',
    method: 'Compares motor readings for each movement with learned patterns.',
    limitation: 'A Normal result means unusual resistance was not detected. It does not confirm that the door is fault-free.',
  },
  acv: {
    name: 'Air conditioning', short: 'ACV', extension: '.xlsx',
    subtitle: 'Prioritise cars for leak checks',
    title: 'See which car to inspect first.',
    description: 'Cars are ranked by how strongly their cooling readings differ from their peers.',
    output: 'Car inspection order',
    action: 'Start with the first car, then continue down the ranking if needed.',
    hint: 'Add Excel files containing readings for every car.',
    format: 'Excel workbooks with the original car-specific column headings and readings.',
    method: 'Compares temperatures across cars and uses pressure readings when available.',
    limitation: 'The ranking is an inspection order, not a confirmed leak or probability of failure.',
  },
  rail: {
    name: 'Rail condition', short: 'Rail', extension: '.csv',
    subtitle: 'Locate possible corrugation',
    title: 'Find which rail side needs inspection.',
    description: 'Vibration measurements are checked for repeated uneven rail wear on Side I or Side II.',
    output: 'Condition for each recording',
    action: 'Use the recording location to inspect the indicated rail side.',
    hint: 'Add one-second vibration recordings.',
    format: 'CSV files with one speed signal and 128 original vibration and shock columns.',
    method: 'Adjusts vibration measurements for speed and compares learned wear patterns by side.',
    limitation: 'A Normal result does not rule out every rail defect. Confirm the recording location before inspection.',
  },
  shm: {
    name: 'Structural health', short: 'SHM', extension: '.csv',
    subtitle: 'Compare fatigue damage',
    title: 'Compare damage from repeated stress.',
    description: 'Stress cycles are converted into a fatigue-damage estimate for each recording.',
    output: 'Estimated damage by file',
    action: 'Review higher values among recordings from comparable parts and periods.',
    hint: 'Add headerless stress recordings.',
    format: 'Headerless CSV files with numeric stress readings in the first column.',
    method: 'Counts stress cycles, estimates fatigue damage, and applies a learned correction.',
    limitation: 'The estimate is not remaining life. Compare only equivalent assets and recording periods.',
  },
};

const statusLabels = {
  new: 'New',
  acknowledged: 'Acknowledged',
  inspection_scheduled: 'Inspection scheduled',
  resolved: 'Resolved',
  false_alert: 'False alert',
};

const state = {
  system: 'rail',
  files: {door: [], acv: [], rail: [], shm: []},
  contexts: {door: {}, acv: {}, rail: {}, shm: {}},
  models: {},
  runs: [],
  current: null,
  busy: false,
  view: 'workspace',
  selections: {},
};

let toastTimer;

function toast(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4200);
}

function showError(message) {
  $('#error').textContent = message;
  $('#error').hidden = !message;
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The request could not be completed.');
  return data;
}

function currentFiles() {
  return state.files[state.system];
}

function readContext() {
  state.contexts[state.system] = {
    asset_id: $('#asset-id').value.trim(),
    location: $('#asset-location').value.trim(),
    collected_at: $('#collected-at').value.trim(),
    work_order: $('#work-order').value.trim(),
  };
  return state.contexts[state.system];
}

function writeContext() {
  const context = state.contexts[state.system];
  $('#asset-id').value = context.asset_id || '';
  $('#asset-location').value = context.location || '';
  $('#collected-at').value = context.collected_at || '';
  $('#work-order').value = context.work_order || '';
}

function openReviews() {
  return state.runs.filter((run) => !run.preview && !['resolved', 'false_alert'].includes(run.review?.status));
}

function updateCounts() {
  $('#history-count').textContent = openReviews().length;
  $('#export-count').textContent = `${Object.keys(latestLive()).length} / 4`;
}

function setView(view) {
  state.view = view;
  window.scrollTo({top: 0, behavior: 'instant'});
  document.querySelectorAll('.view').forEach((element) => {
    element.hidden = element.id !== `${view}-view`;
  });
  document.querySelectorAll('.nav').forEach((element) => {
    element.classList.toggle('active', element.dataset.view === view);
  });
  $('#page-name').textContent = {
    workspace: 'New condition check',
    history: 'Review queue',
    guide: 'Help and system status',
  }[view];
  if (view === 'history') renderHistory();
  if (view === 'guide') renderModels();
}

function selectSystem(key) {
  if (state.busy) return;
  readContext();
  state.system = key;
  state.current = null;
  $('#results').hidden = true;
  showError('');
  renderWorkspace();
}

function renderWorkspace() {
  const system = systems[state.system];
  $('#system-cards').innerHTML = Object.entries(systems).map(([key, item]) => `
    <button class="system-card ${key === state.system ? 'selected' : ''}" data-system="${key}"
      aria-pressed="${key === state.system}" ${state.busy ? 'disabled' : ''}>
      <span class="system-icon"><svg viewBox="0 0 24 24" aria-hidden="true">${icons[key]}</svg></span>
      <span><strong>${item.name}</strong><small>${item.subtitle}</small></span>
    </button>
  `).join('');
  $('#system-cards').querySelectorAll('button').forEach((button) => {
    button.onclick = () => selectSystem(button.dataset.system);
  });
  $('#input-type').textContent = system.extension.slice(1).toUpperCase();
  $('#input-hint').textContent = system.hint;
  $('#file-input').accept = system.extension;
  $('#file-input').multiple = state.system !== 'door';
  $('#file-input').disabled = state.busy;
  $('#upload-limit').textContent = `${system.extension.slice(1).toUpperCase()} · ${state.system === 'door' ? 'one recording' : 'up to 100 files'} · 120 MB total`;
  $('#context-title').textContent = system.title;
  $('#context-description').textContent = system.description;
  $('#asset-tag').textContent = system.short.toUpperCase();
  $('#output-summary').textContent = system.output;
  $('#action-summary').textContent = system.action;
  writeContext();
  renderFiles();
  renderNotice();
}

function renderNotice() {
  const model = state.models[state.system];
  const notice = $('#model-notice');
  notice.className = `notice ${model?.ready ? 'ready' : ''}`;
  notice.textContent = model
    ? (model.ready ? '✓ Ready to check files' : 'This check is unavailable. Open Help to review system status.')
    : 'Checking system status…';
  $('#analyze').disabled = state.busy || !currentFiles().length || !model?.ready;
  $('#analyze').innerHTML = state.busy ? 'Checking files…' : 'Check files <span aria-hidden="true">→</span>';
  $('#preview').disabled = state.busy;
  $('#progress').hidden = !state.busy;
}

async function refreshStatus() {
  try {
    state.models = (await api('/api/status')).models;
    renderNotice();
    if (state.view === 'guide') renderModels();
  } catch (error) {
    $('#model-notice').textContent = 'The app has lost its connection. Restart it and refresh this page.';
    showError(error.message);
  }
}

function fileSize(file) {
  return file.size < 1048576
    ? `${(file.size / 1024).toFixed(0)} KB`
    : `${(file.size / 1048576).toFixed(1)} MB`;
}

function renderFiles() {
  $('#file-list').innerHTML = currentFiles().map((file, index) => `
    <div class="file-item">
      <span title="${esc(file.name)}">${esc(file.name)}</span>
      <small><i class="ready-dot"></i> Selected · ${fileSize(file)}</small>
      <button class="icon-button" aria-label="Remove ${esc(file.name)}" data-index="${index}" ${state.busy ? 'disabled' : ''}>×</button>
    </div>
  `).join('');
  $('#file-list').querySelectorAll('button').forEach((button) => {
    button.onclick = () => {
      currentFiles().splice(Number(button.dataset.index), 1);
      renderFiles();
      renderNotice();
    };
  });
}

function addFiles(files) {
  if (state.busy) return;
  showError('');
  const next = [...currentFiles(), ...files];
  const system = systems[state.system];
  if (next.some((file) => !file.name.endsWith(system.extension))) {
    showError(`Choose ${system.extension} files for ${system.name}.`);
    return;
  }
  if (next.some((file) => !file.size)) {
    showError('One file is empty. Choose a recording containing sensor data.');
    return;
  }
  if (new Set(next.map((file) => file.name.toLowerCase())).size !== next.length) {
    showError('A filename is repeated. Remove the duplicate before continuing.');
    return;
  }
  if (next.length > (state.system === 'door' ? 1 : 100)) {
    showError(state.system === 'door' ? 'Choose one continuous door recording.' : 'Choose up to 100 files at a time.');
    return;
  }
  if (next.reduce((total, file) => total + file.size, 0) > 120 * 1048576) {
    showError('These files exceed 120 MB. Check them in smaller groups.');
    return;
  }
  state.files[state.system] = next;
  renderFiles();
  renderNotice();
}

function addRun(run) {
  state.runs = [run, ...state.runs.filter((item) => item.id !== run.id)].slice(0, 500);
  state.current = run;
  ensureSelection(run);
  updateCounts();
  renderResults(run);
}

function latestLive() {
  const latest = {};
  for (const run of state.runs) {
    if (run.preview) continue;
    if (!latest[run.system]) latest[run.system] = {rows: [], names: new Set()};
    const group = latest[run.system];
    if (run.system === 'door') {
      if (!group.rows.length) group.rows = run.rows;
    } else {
      for (const row of run.rows) {
        if (!group.names.has(row.file_id)) {
          group.names.add(row.file_id);
          group.rows.push(row);
        }
      }
    }
  }
  return latest;
}

async function analyze() {
  if (state.busy) return;
  state.busy = true;
  state.current = null;
  $('#results').hidden = true;
  showError('');
  const data = new FormData();
  data.append('system', state.system);
  currentFiles().forEach((file) => data.append('files', file, file.name));
  Object.entries(readContext()).forEach(([key, value]) => data.append(key, value));
  renderWorkspace();
  try {
    const run = await api('/api/analyze', {method: 'POST', body: data});
    addRun(run);
    toast(run.failures.length ? 'Check complete. Some files need correction.' : 'Check complete. Results are ready.');
    $('#results').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    renderWorkspace();
    refreshStatus();
  }
}

async function preview() {
  if (state.busy) return;
  const key = state.system;
  showError('');
  $('#preview').disabled = true;
  try {
    const run = await api(`/api/preview?system=${key}`);
    if (key === state.system) {
      addRun(run);
      $('#results').scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  } catch (error) {
    showError(error.message);
  } finally {
    $('#preview').disabled = false;
  }
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
}

function summary(run) {
  const rows = run.rows;
  if (run.system === 'rail' || run.system === 'door') {
    const flagged = rows.filter((row) => row.prediction !== 'Normal').length;
    return [
      {label: run.system === 'door' ? 'Movements' : 'Recordings', value: rows.length},
      {label: 'Needs inspection', value: flagged},
      {label: 'No target fault detected', value: rows.length - flagged},
    ];
  }
  if (run.system === 'acv') {
    const first = rows[0].ranked_cars.split('|');
    return [
      {label: 'Recordings', value: rows.length},
      {label: 'Inspect first', value: `Car ${first[0]}`},
      {label: 'Cars ranked', value: first.length},
    ];
  }
  const values = rows.map((row) => Number(row.prediction));
  return [
    {label: 'Recordings', value: rows.length},
    {label: 'Highest estimate', value: Math.max(...values).toPrecision(4)},
    {label: 'Batch median', value: median(values).toPrecision(4)},
  ];
}

function insight(run) {
  const rows = run.rows;
  if (run.system === 'rail') {
    const sideI = rows.filter((row) => row.prediction === 'Side I').length;
    const sideII = rows.filter((row) => row.prediction === 'Side II').length;
    return sideI + sideII
      ? `${sideI + sideII} recording(s) need review: ${sideI} on Side I and ${sideII} on Side II.`
      : 'No corrugation target was detected. Continue routine inspection.';
  }
  if (run.system === 'door') {
    const flagged = rows.filter((row) => row.prediction !== 'Normal').length;
    return flagged
      ? `${flagged} movement(s) show unusual resistance. Match their times to the door operation.`
      : 'No unusual resistance was detected. Confirm that every movement was found.';
  }
  if (run.system === 'acv') {
    return 'Inspect the first car in each ranking, then continue down the list if the first check is inconclusive.';
  }
  const highest = rows.reduce((left, right) => Number(left.prediction) > Number(right.prediction) ? left : right);
  return `${highest.file_id} has the highest estimate. Compare only equivalent parts and recording periods.`;
}

function defaultSelection(run) {
  if (run.system === 'door') {
    const index = run.rows.findIndex((row) => row.prediction !== 'Normal');
    return {movement: index >= 0 ? index + 1 : 1};
  }
  if (run.system === 'acv') {
    const row = run.rows[0];
    return {file: row.file_id, car: row.ranked_cars.split('|')[0]};
  }
  if (run.system === 'rail') {
    const row = run.rows.find((item) => item.prediction !== 'Normal') || run.rows[0];
    return {file: row.file_id};
  }
  const row = run.rows.reduce((left, right) => Number(left.prediction) > Number(right.prediction) ? left : right);
  return {file: row.file_id};
}

function ensureSelection(run) {
  if (!state.selections[run.id]) state.selections[run.id] = defaultSelection(run);
  return state.selections[run.id];
}

function isSelected(run, type, value) {
  return String(ensureSelection(run)[type]) === String(value);
}

function chart(run) {
  const rows = run.rows;
  if (run.system === 'rail') {
    return `
      <div class="chart-title"><strong>Condition overview</strong><span>Select a file below for evidence</span></div>
      <div class="distribution" aria-label="Rail result distribution">
        ${['Normal', 'Side I', 'Side II'].map((label, index) => {
          const count = rows.filter((row) => row.prediction === label).length;
          return count ? `<div class="distribution-${index}" style="width:${count / rows.length * 100}%">${label}: ${count}</div>` : '';
        }).join('')}
      </div>
    `;
  }
  if (run.system === 'door') {
    return `
      <div class="chart-title"><strong>Movements in recording order</strong><span>Select a movement for evidence</span></div>
      <div class="segments">
        ${rows.map((row, index) => `
          <button class="segment ${row.prediction !== 'Normal' ? 'flag' : ''} ${isSelected(run, 'movement', index + 1) ? 'active' : ''}"
            data-evidence-movement="${index + 1}" aria-label="Movement ${index + 1}: ${esc(row.prediction)}">
            ${String(index + 1).padStart(2, '0')}
          </button>
        `).join('')}
      </div>
    `;
  }
  if (run.system === 'acv') {
    const selected = ensureSelection(run);
    const row = rows.find((item) => item.file_id === selected.file) || rows[0];
    return `
      <div class="chart-title">
        <strong>Car inspection order</strong>
        <select id="acv-case" aria-label="Choose an air-conditioning recording">
          ${rows.map((item) => `<option value="${esc(item.file_id)}" ${item.file_id === row.file_id ? 'selected' : ''}>${esc(item.file_id)}</option>`).join('')}
        </select>
      </div>
      <div class="ranking">
        ${row.ranked_cars.split('|').map((car, index) => `
          <button class="rank-car ${isSelected(run, 'car', car) ? 'active' : ''}" data-evidence-car="${esc(car)}">
            <small>${index === 0 ? 'INSPECT FIRST' : `RANK ${index + 1}`}</small><strong>${esc(car)}</strong><span>CAR</span>
          </button>
        `).join('')}
      </div>
    `;
  }
  const sorted = [...rows].sort((left, right) => Number(right.prediction) - Number(left.prediction)).slice(0, 24);
  const maximum = Math.max(...sorted.map((row) => Number(row.prediction)), 0.000001);
  return `
    <div class="chart-title"><strong>Damage estimates · highest first</strong><span>Select a bar for evidence</span></div>
    <div class="bars">
      ${sorted.map((row, index) => `
        <button class="bar-column ${isSelected(run, 'file', row.file_id) ? 'active' : ''}"
          data-evidence-file="${esc(row.file_id)}" style="height:${Number(row.prediction) / maximum * 100}%"
          aria-label="${esc(row.file_id)}: ${Number(row.prediction).toPrecision(6)}">
          <span>${index + 1}</span>
        </button>
      `).join('')}
    </div>
  `;
}

function statusText(status) {
  return {
    only: 'No peer comparison',
    typical: 'Within batch range',
    above: 'Above batch average',
    below: 'Below batch average',
    outlier_high: 'High outlier',
    outlier_low: 'Low outlier',
  }[status] || status;
}

function formatMetric(value, unit) {
  return `${Number(value).toLocaleString(undefined, {maximumFractionDigits: 6})}${unit ? ` ${unit}` : ''}`;
}

function comparisonCards(comparisons) {
  if (!comparisons?.length) return '<p class="muted">No peer comparison is available for this item.</p>';
  return `<div class="comparison-grid">${comparisons.map((item) => `
    <div class="comparison ${esc(item.status)}">
      <small>${esc(item.label)}</small>
      <strong>${formatMetric(item.value, item.unit)}</strong>
      <span>${esc(statusText(item.status))}</span>
      <p>Batch average ${formatMetric(item.average, item.unit)} · ${item.peer_count} item${item.peer_count === 1 ? '' : 's'}</p>
    </div>
  `).join('')}</div>`;
}

function evidencePanel(run) {
  if (run.preview || !run.evidence || !Object.keys(run.evidence).length) {
    return `
      <section class="evidence-panel">
        <h3>Why this result?</h3>
        <p class="muted">Evidence is shown after checking original sensor files. Saved examples contain predictions only.</p>
      </section>
    `;
  }
  const selected = ensureSelection(run);
  let title = '';
  let details = '';
  let comparisons = [];
  if (run.system === 'door') {
    const item = run.evidence.items?.find((entry) => entry.movement === Number(selected.movement));
    if (!item) return '';
    title = `Movement ${item.movement} · ${item.operation}`;
    details = `Model score ${item.model_score} · decision threshold ${item.decision_threshold} · ${item.duration_seconds} s`;
    comparisons = item.comparisons;
  } else if (run.system === 'acv') {
    const file = run.evidence.files?.[selected.file];
    const item = file?.items?.find((entry) => entry.car === selected.car);
    if (!file || !item) return '';
    title = `${selected.file} · Car ${selected.car}`;
    details = selected.car === file.top_car
      ? `Ranked first · score gap to Car ${file.runner_up}: ${file.score_gap ?? 'not available'}`
      : `Compare this car with the other cars in the same recording.`;
    comparisons = item.comparisons;
  } else {
    const item = run.evidence.files?.[selected.file];
    if (!item) return '';
    title = selected.file;
    if (run.system === 'rail') {
      details = `Side I score ${item.side_i_score} · Side II score ${item.side_ii_score} · threshold ${item.decision_threshold} · ${item.speed_mps} m/s`;
    } else {
      details = `${item.samples?.toLocaleString()} samples · ${item.counted_cycles?.toLocaleString()} counted cycles`;
    }
    comparisons = item.comparisons;
  }
  return `
    <section class="evidence-panel">
      <div class="evidence-heading">
        <div><span class="eyebrow">SELECTED ITEM</span><h3>${esc(title)}</h3></div>
        <span class="evidence-note">Compared only with this recording or batch</span>
      </div>
      <p class="decision-detail">${esc(details)}</p>
      ${comparisonCards(comparisons)}
      <p class="data-note">Relative colour highlights unusual measurements; it is not a confirmed fault or fleet threshold.</p>
    </section>
  `;
}

function contextLine(run) {
  const values = [run.context?.asset_id, run.context?.location, run.context?.work_order].filter(Boolean);
  return values.length ? values.join(' · ') : 'Asset details not recorded';
}

function previousAssetText(run) {
  const asset = run.context?.asset_id;
  if (!asset || run.preview) return '';
  const previous = state.runs.filter((item) => item.id !== run.id && !item.preview
    && item.system === run.system && item.context?.asset_id?.toLowerCase() === asset.toLowerCase());
  if (!previous.length) return '';
  if (run.system === 'shm') {
    const current = Math.max(...run.rows.map((row) => Number(row.prediction)));
    const earlier = Math.max(...previous[0].rows.map((row) => Number(row.prediction)));
    const change = earlier ? (current - earlier) / earlier * 100 : 0;
    return `<div class="trend-note"><strong>Previous check:</strong> highest estimate ${change >= 0 ? 'increased' : 'decreased'} ${Math.abs(change).toFixed(1)}% since ${new Date(previous[0].created).toLocaleDateString()}.</div>`;
  }
  return `<div class="trend-note"><strong>History:</strong> ${previous.length} earlier ${systems[run.system].name.toLowerCase()} check${previous.length === 1 ? '' : 's'} found for this asset.</div>`;
}

function failurePanel(run) {
  if (!run.failures?.length) return '';
  return `
    <div class="batch-warning">
      <strong>${run.failures.length} file${run.failures.length === 1 ? '' : 's'} could not be checked</strong>
      ${run.failures.map((failure) => `<p><b>${esc(failure.name)}</b> · ${esc(failure.error)}</p>`).join('')}
    </div>
  `;
}

function resultTable(run) {
  const selected = ensureSelection(run);
  let headers;
  if (run.system === 'door') headers = ['Movement', 'Start', 'End', 'Result'];
  else if (run.system === 'acv') headers = ['Source file', 'Inspection order'];
  else headers = ['Source file', run.system === 'shm' ? 'Damage estimate' : 'Result'];
  const body = run.rows.slice(0, 500).map((row, index) => {
    let cells;
    let attributes = '';
    if (run.system === 'door') {
      const movement = index + 1;
      attributes = `data-evidence-movement="${movement}" tabindex="0" role="button" class="selectable-row ${String(selected.movement) === String(movement) ? 'active' : ''}"`;
      cells = [String(movement).padStart(2, '0'), esc(row.start_time), esc(row.end_time), pill(row.prediction)];
    } else if (run.system === 'acv') {
      attributes = `data-evidence-file="${esc(row.file_id)}" tabindex="0" role="button" class="selectable-row ${selected.file === row.file_id ? 'active' : ''}"`;
      cells = [esc(row.file_id), esc(row.ranked_cars.split('|').join(' → '))];
    } else {
      attributes = `data-evidence-file="${esc(row.file_id)}" tabindex="0" role="button" class="selectable-row ${selected.file === row.file_id ? 'active' : ''}"`;
      cells = [esc(row.file_id), run.system === 'shm' ? Number(row.prediction).toPrecision(7) : pill(row.prediction)];
    }
    return `<tr ${attributes}>${cells.map((cell) => `<td>${cell}</td>`).join('')}</tr>`;
  }).join('');
  return `
    <div class="table-tools"><h3>All results <span class="muted">${run.rows.length}</span></h3></div>
    <div class="table-wrap"><table><thead><tr>${headers.map((header) => `<th scope="col">${header}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>
  `;
}

function pill(value) {
  return `<span class="pill ${value !== 'Normal' ? 'flag' : ''}">${esc(value)}</span>`;
}

function reviewForm(run) {
  if (run.preview) return '';
  const review = run.review || {status: 'new', note: ''};
  return `
    <form id="review-form" class="review-form">
      <div>
        <span class="eyebrow">INSPECTION FOLLOW-UP</span>
        <h3>Record what happens next</h3>
      </div>
      <label>Status
        <select id="review-status">
          ${Object.entries(statusLabels).map(([value, label]) => `<option value="${value}" ${review.status === value ? 'selected' : ''}>${label}</option>`).join('')}
        </select>
      </label>
      <label>Technician note
        <textarea id="review-note" maxlength="2000" rows="3" placeholder="Inspection finding or next action">${esc(review.note)}</textarea>
      </label>
      <button class="button primary" type="submit">Save follow-up</button>
    </form>
  `;
}

function renderResults(run) {
  const system = systems[run.system];
  ensureSelection(run);
  $('#results').hidden = false;
  $('#results').innerHTML = `
    <div class="results-header">
      <div>
        <span class="result-badge ${run.preview ? '' : 'live'}">${run.preview ? 'EXAMPLE · NO FILES CHECKED' : 'CHECK COMPLETE'}</span>
        <h2>${esc(contextLine(run))}</h2>
        <p class="result-meta">${system.name}${run.preview ? '' : ` · ${esc(run.model?.run_id || 'Active model')} · ${run.elapsed.toFixed(2)} s`}</p>
      </div>
      <div class="results-actions">
        <a class="button secondary" href="/api/runs/${run.id}/inspection">Inspection report</a>
        <a class="button primary" href="/api/runs/${run.id}/csv">Download results</a>
      </div>
    </div>
    ${failurePanel(run)}
    <div class="stat-grid">
      ${summary(run).map((item) => `<div class="stat"><small>${item.label}</small><strong>${esc(item.value)}</strong></div>`).join('')}
    </div>
    <div class="insight"><span aria-hidden="true">→</span><div><strong>What to check next</strong><p>${esc(insight(run))}</p></div></div>
    ${previousAssetText(run)}
    ${chart(run)}
    ${evidencePanel(run)}
    ${resultTable(run)}
    <p class="data-note">${esc(system.limitation)}</p>
    ${reviewForm(run)}
  `;
  bindResultEvents(run);
}

function bindResultEvents(run) {
  const makeKeyboardClickable = (element) => {
    element.onkeydown = (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        element.click();
      }
    };
  };
  $('#results').querySelectorAll('[data-evidence-movement]').forEach((element) => {
    makeKeyboardClickable(element);
    element.onclick = () => {
      state.selections[run.id] = {movement: Number(element.dataset.evidenceMovement)};
      renderResults(run);
    };
  });
  $('#results').querySelectorAll('[data-evidence-file]').forEach((element) => {
    makeKeyboardClickable(element);
    element.onclick = () => {
      const selection = ensureSelection(run);
      selection.file = element.dataset.evidenceFile;
      if (run.system === 'acv') {
        const row = run.rows.find((item) => item.file_id === selection.file);
        selection.car = row.ranked_cars.split('|')[0];
      }
      renderResults(run);
    };
  });
  $('#results').querySelectorAll('[data-evidence-car]').forEach((element) => {
    makeKeyboardClickable(element);
    element.onclick = () => {
      ensureSelection(run).car = element.dataset.evidenceCar;
      renderResults(run);
    };
  });
  if ($('#acv-case')) {
    $('#acv-case').onchange = (event) => {
      const row = run.rows.find((item) => item.file_id === event.target.value);
      state.selections[run.id] = {file: row.file_id, car: row.ranked_cars.split('|')[0]};
      renderResults(run);
    };
  }
  if ($('#review-form')) {
    $('#review-form').onsubmit = async (event) => {
      event.preventDefault();
      const button = event.currentTarget.querySelector('button');
      button.disabled = true;
      try {
        const updated = await api(`/api/runs/${run.id}/review`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({status: $('#review-status').value, note: $('#review-note').value}),
        });
        state.runs = state.runs.map((item) => item.id === updated.id ? updated : item);
        state.current = updated;
        updateCounts();
        renderResults(updated);
        toast('Follow-up saved.');
      } catch (error) {
        toast(error.message);
        button.disabled = false;
      }
    };
  }
}

function outcomeText(run) {
  if (run.system === 'door' || run.system === 'rail') {
    const flagged = run.rows.filter((row) => row.prediction !== 'Normal').length;
    return flagged ? `${flagged} need inspection` : 'No target fault detected';
  }
  if (run.system === 'acv') return `Inspect Car ${run.rows[0].ranked_cars.split('|')[0]} first`;
  return `Highest estimate ${Math.max(...run.rows.map((row) => Number(row.prediction))).toPrecision(4)}`;
}

function historyMatches(run, query) {
  return JSON.stringify({context: run.context, files: run.files, rows: run.rows})
    .toLowerCase().includes(query.toLowerCase());
}

function renderHistory() {
  const query = $('#history-search').value;
  const filter = $('#history-filter').value;
  const runs = state.runs.filter((run) => {
    if (run.preview) return filter === 'all' && historyMatches(run, query);
    const status = run.review?.status || 'new';
    const statusMatch = filter === 'all' || (filter === 'open' && !['resolved', 'false_alert'].includes(status)) || filter === status;
    return statusMatch && historyMatches(run, query);
  });
  $('#history-content').innerHTML = runs.length ? runs.map((run) => `
    <article class="history-item">
      <span class="status-pill status-${esc(run.review?.status || 'example')}">${run.preview ? 'Example' : esc(statusLabels[run.review?.status] || 'New')}</span>
      <div>
        <h3>${esc(run.context?.asset_id || `${systems[run.system].name} check`)}</h3>
        <p>${esc([run.context?.location, run.context?.work_order, new Date(run.created).toLocaleString()].filter(Boolean).join(' · '))}</p>
        <strong>${esc(outcomeText(run))}</strong>
      </div>
      <button class="button secondary" data-run="${run.id}">Open analysis →</button>
    </article>
  `).join('') : '<div class="panel empty"><h2>No matching checks.</h2><p>Change the filter or start a new condition check.</p></div>';
  $('#history-content').querySelectorAll('[data-run]').forEach((button) => {
    button.onclick = () => {
      const run = state.runs.find((item) => item.id === button.dataset.run);
      state.system = run.system;
      state.current = run;
      setView('workspace');
      renderWorkspace();
      renderResults(run);
      $('#results').scrollIntoView({behavior: 'smooth'});
    };
  });
}

function renderModels() {
  $('#model-content').innerHTML = Object.entries(systems).map(([key, system]) => {
    const model = state.models[key];
    return `
      <article class="model-card ${model?.ready ? 'ready' : 'not-ready'}">
        <div><strong>${system.name}</strong><span>${model?.ready ? 'Ready' : 'Needs setup'}</span></div>
        <p>${esc(model?.message || 'Checking status…')}</p>
        <details><summary>Technical details</summary>
          <p>${esc(system.method)}</p>
          <code>${esc(model?.run_id || 'Model version unavailable')}</code>
        </details>
      </article>
    `;
  }).join('');
}

function openExport() {
  const latest = latestLive();
  $('#export-checklist').innerHTML = Object.entries(systems).map(([key, system]) => `
    <div class="check-row"><span>${latest[key] ? '✓' : '○'}</span><strong>${system.name}</strong><small>${latest[key] ? `${latest[key].rows.length} results` : 'No completed check'}</small></div>
  `).join('');
  $('#download-zip').disabled = !Object.keys(latest).length;
  $('#export-error').hidden = true;
  $('#export-dialog').showModal();
}

async function downloadZip() {
  const button = $('#download-zip');
  button.disabled = true;
  try {
    const response = await fetch('/api/export', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ids: state.runs.filter((run) => !run.preview).map((run) => run.id)}),
    });
    if (!response.ok) throw new Error((await response.json()).error);
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'predictions.zip';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    $('#export-dialog').close();
    toast('Submission ZIP downloaded.');
  } catch (error) {
    $('#export-error').textContent = error.message;
    $('#export-error').hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function restoreHistory() {
  try {
    state.runs = (await api('/api/history')).runs;
    updateCounts();
    if (state.view === 'history') renderHistory();
  } catch (error) {
    toast(`History could not be restored: ${error.message}`);
  }
}

document.querySelectorAll('.nav').forEach((button) => {
  button.onclick = () => setView(button.dataset.view);
});
$('#file-input').onchange = (event) => {
  addFiles([...event.target.files]);
  event.target.value = '';
};
const dropzone = $('#dropzone');
['dragenter', 'dragover'].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragover');
}));
dropzone.addEventListener('drop', (event) => addFiles([...event.dataTransfer.files]));
$('#analyze').onclick = analyze;
$('#preview').onclick = preview;
$('#refresh-models').onclick = refreshStatus;
$('#export-open').onclick = openExport;
$('#close-export').onclick = () => $('#export-dialog').close();
$('#download-zip').onclick = downloadZip;
$('#history-search').oninput = renderHistory;
$('#history-filter').onchange = renderHistory;
$('.train-cars').innerHTML = Array.from({length: 8}, (_, index) => `
  <div class="car"><i></i><i></i><i></i><b>${String(index + 1).padStart(2, '0')}</b></div>
`).join('');
$('#format-guide').innerHTML = Object.values(systems).map((system) => `<h3>${system.name}</h3><p>${system.format}</p>`).join('');

renderWorkspace();
refreshStatus();
restoreHistory();
