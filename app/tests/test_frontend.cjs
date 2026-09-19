const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
function harness(){
 const nodes=new Map();const node=()=>({innerHTML:'',textContent:'',hidden:false,value:'',scrollIntoView(){},append(){},querySelectorAll(){return[];}});
 const context=vm.createContext({console,FormData:class{append(){}},setTimeout,clearTimeout,document:{querySelector(s){if(!nodes.has(s))nodes.set(s,node());return nodes.get(s);}},localStorage:{getItem(){return JSON.stringify({system:'door'});},setItem(){}},window:{scrollTo(){}}});
 let src=fs.readFileSync('app/static/app.js','utf8').split("document.querySelectorAll('.nav').forEach(button=>")[0];
 vm.runInContext(src,context);
 vm.runInContext(`globalThis.shown=[];globalThis.responses=[];globalThis.calls=[];renderWorkspace=()=>{};renderNotice=()=>{};refreshStatus=()=>{};updateCounts=()=>{};toast=()=>{};renderResults=r=>shown.push(r.id);api=async(url)=>{calls.push(url);return responses.shift()();};`,context);
 return {context,nodes,run:s=>vm.runInContext(s,context)};
}
test('batch results stay hidden until all files and the merge finish',async()=>{
 const h=harness();let release;
 h.context.first={id:'one',system:'rail',files:[{name:'a.csv'}],rows:[]};
 h.context.second={id:'two',system:'rail',files:[{name:'b.csv'}],rows:[]};
 h.context.merged={id:'merged',system:'rail',files:[{name:'a.csv'},{name:'b.csv'}],rows:[],combined_from:['one','two']};
 h.context.responses=[()=>h.context.first,()=>new Promise(r=>release=r),()=>h.context.merged];
 h.run(`state.system='rail';state.files=[{name:'a.csv'},{name:'b.csv'}];state.bySystem.rail={id:'old'};`);
 const pending=h.run('analyze()');await new Promise(r=>setImmediate(r));
 assert.deepEqual(Array.from(h.context.shown),[]);assert.match(h.nodes.get('#results').innerHTML,/Checking recordings/);
 release(h.context.second);await pending;
 assert.deepEqual(Array.from(h.context.shown),['merged']);assert.equal(h.run('state.busy'),false);
});
test('partial batch failure keeps completed checks and hides partial results',async()=>{
 const h=harness();h.context.responses=[()=>({id:'one',system:'rail',files:[{name:'a.csv'}],rows:[]}),()=>{throw new Error('Bad file');}];
 h.run(`state.system='rail';state.files=[{name:'a.csv'},{name:'b.csv'}];`);await h.run('analyze()');
 assert.deepEqual(Array.from(h.context.shown),[]);assert.equal(h.run('state.files[0].name'),'b.csv');assert.match(h.nodes.get('#results').innerHTML,/Check incomplete/);assert.equal(h.run('historyRuns().length'),1);
});
test('saved subsystem and second-only recording timestamps',()=>{
 const h=harness();assert.equal(h.run('state.system'),'door');assert.equal(h.run("displayTime('2023-7-5-0-11-17-664')"),'05 Jul 2023, 00:11:17');
});
test('deviation is percentage of one SD, with neutral and missing cases',()=>{
 const h=harness();vm.runInContext(fs.readFileSync('app/static/technician.js','utf8'),h.context);
 assert.equal(h.run('deviation({value:13,mean:10,low:6,high:14,n:5})'),150);
 assert.equal(h.run('deviation({value:7,mean:10,low:6,high:14,n:5})'),-150);
 assert.equal(h.run('deviation({value:10,mean:10,low:10,high:10,n:5})'),0);
 assert.equal(h.run('deviation({value:13,mean:10,low:null,high:null,n:2})'),null);
 assert.match(h.run('deviationCell({value:11.5,mean:10,low:6,high:14,n:5})'),/rgb\(255,255,255\)/);
 assert.match(h.run('deviationCell({value:13,mean:10,low:6,high:14,n:5})'),/rgb\(206,152,152\)/);
 assert.match(h.run('deviationCell({value:7,mean:10,low:6,high:14,n:5})'),/rgb\(206,152,152\)/);
 assert.match(h.run('deviationCell({value:16,mean:10,low:6,high:14,n:5})'),/rgb\(157,48,48\)/);
 assert.match(h.run('deviationCell({value:10,mean:10,low:6,high:14,n:5})'),/rgb\(35,98,70\)/);
});
