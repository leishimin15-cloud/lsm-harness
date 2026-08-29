const TOKEN=document.querySelector('meta[name="lsm-web-token"]').content;
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const S={route:location.hash.slice(1)||'overview',boot:null,topology:null,turns:[],events:[],cursor:0,seen:new Set(),nodeStates:new Map(),activeTurn:'',busy:false,replay:null,timer:null};

// Small dependency-free Markdown renderer for model replies. Model output is
// escaped before any formatting is applied, so it cannot inject page markup.
function mdInline(value){
  return value
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*]+?)\*\*/g,'<strong>$1</strong>')
    .replace(/(^|[^*_`])[*_]([^*_`\s][^*_`]*?)[*_](?![\w*])/g,'$1<em>$2</em>')
    .replace(/`([^`]+?)`/g,'<code>$1</code>');
}
function renderMarkdown(text){
  const lines=esc(text).split(/\r?\n/),out=[];
  const tableRow=line=>/^\s*\|.*\|\s*$/.test(line);
  const tableSep=line=>/^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(line);
  const cells=line=>line.trim().replace(/^\||\|$/g,'').split('|').map(cell=>cell.trim());
  const special=(line,index)=>/^\s*`{3,}/.test(line)||/^\s*#{1,6}\s+/.test(line)||/^\s*[-*]\s+/.test(line)||/^\s*\d+\.\s+/.test(line)||/^\s*[-*_]{3,}\s*$/.test(line)||(tableRow(line)&&index+1<lines.length&&tableSep(lines[index+1]));
  let i=0;
  while(i<lines.length){
    const line=lines[i];
    if(/^\s*`{3,}/.test(line)){
      const lang=line.replace(/^\s*`{3,}/,'').trim();i++;
      const code=[];while(i<lines.length&&!/^\s*`{3,}\s*$/.test(lines[i]))code.push(lines[i++]);
      if(i<lines.length)i++;
      out.push(`<div class="mdcode">${lang?`<div class="mdcode-head">${lang}</div>`:''}<pre><code>${code.join('\n')}</code></pre></div>`);continue;
    }
    if(tableRow(line)&&i+1<lines.length&&tableSep(lines[i+1])){
      const head=cells(line),body=[];i+=2;while(i<lines.length&&tableRow(lines[i]))body.push(cells(lines[i++]));
      out.push(`<div class="mdtable-wrap"><table class="mdtable"><thead><tr>${head.map(cell=>`<th>${mdInline(cell)}</th>`).join('')}</tr></thead><tbody>${body.map(row=>`<tr>${row.map(cell=>`<td>${mdInline(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);continue;
    }
    const heading=line.match(/^\s*(#{1,6})\s+(.*)$/);
    if(heading){out.push(`<div class="mdh mdh-${heading[1].length}">${mdInline(heading[2])}</div>`);i++;continue}
    if(/^\s*[-*]\s+/.test(line)){
      const items=[];while(i<lines.length&&/^\s*[-*]\s+/.test(lines[i]))items.push(mdInline(lines[i++].replace(/^\s*[-*]\s+/,'')));
      out.push(`<ul class="mdlist">${items.map(item=>`<li>${item}</li>`).join('')}</ul>`);continue;
    }
    if(/^\s*\d+\.\s+/.test(line)){
      const items=[];while(i<lines.length&&/^\s*\d+\.\s+/.test(lines[i]))items.push(mdInline(lines[i++].replace(/^\s*\d+\.\s+/,'')));
      out.push(`<ol class="mdlist">${items.map(item=>`<li>${item}</li>`).join('')}</ol>`);continue;
    }
    if(/^\s*[-*_]{3,}\s*$/.test(line)){out.push('<hr class="mdhr">');i++;continue}
    if(!line.trim()){i++;continue}
    const para=[];while(i<lines.length&&lines[i].trim()&&!special(lines[i],i))para.push(mdInline(lines[i++]));
    out.push(`<div class="mdp">${para.join('<br>')}</div>`);
  }
  return out.join('');
}

async function api(path,options={}){const headers={'X-LSM-Web-Token':TOKEN,...(options.headers||{})};if(options.body)headers['Content-Type']='application/json';const r=await fetch(path,{...options,headers});if(!r.ok){const d=await r.json().catch(()=>({}));throw new Error(d.error||r.statusText)}return r.json()}
const ms=v=>v==null?'—':v<1000?`${v}ms`:`${(v/1000).toFixed(1)}s`;
function toast(t){const e=$('#toast');e.textContent=t;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),1800)}
function pageHead(t,s){return `<div class="page-head"><div><h1>${esc(t)}</h1><p>${esc(s)}</p></div></div>`}
function metric(k,v){return `<div class="metric"><small>${esc(k)}</small><b>${esc(v)}</b></div>`}
function setRuntime(r){S.busy=!!r.busy;S.activeTurn=r.active_turn_id||S.activeTurn;$('#live-dot').className=`dot ${S.busy?'busy':'live'}`;$('#runtime-state').textContent=S.busy?'trace running':'live';$('#stop-turn').classList.toggle('hidden',!S.busy);$('#steer-row').classList.toggle('hidden',!S.busy);$('#send-message').disabled=S.busy}

async function boot(){const d=await api('/api/bootstrap');S.boot=d;S.topology=d.topology;S.turns=d.turns||[];$('#model-label').textContent=`${d.provider||'provider'} · ${d.model||'model'}`;$('#session-label').textContent=`${d.session_id.slice(0,8)} · ${d.thinking}`;$('#dock-model').textContent=d.model||'model';$('#n-loop').textContent=S.turns.length||'';setRuntime(d.runtime);render();if(!S.polling){S.polling=true;poll()}}
function render(){document.querySelectorAll('#nav button').forEach(b=>b.classList.toggle('active',b.dataset.route===S.route));if(S.route==='overview')overview();else if(S.route==='loop')loop();else dataPage(S.route)}

function overview(){const decisions=S.events.filter(e=>e.type==='memory.gate.decided');const retrieved=decisions.filter(e=>e.data?.decision!=='skip').length;const skipped=decisions.length-retrieved;const total=Math.max(1,decisions.length);$('#main').innerHTML=pageHead('Overview','完整 Agent 架构 · 由 CLI、TUI 与 Web 的真实 Trace 共同点亮')+`<div class="metric-row">${metric('traces',S.turns.length)}${metric('active',S.busy?'1':'0')}${metric('tools',S.events.filter(e=>e.type==='tool.completed').length)}${metric('errors',S.turns.filter(t=>t.status==='error').length)}</div><h2 class="section-title">Retrieval gate — the hero decision</h2><div class="splitbar"><div class="skip" style="width:${skipped/total*100}%">${skipped?`${skipped} skipped`:''}</div><div class="retrieve" style="width:${retrieved/total*100}%">${retrieved?`${retrieved} retrieved`:''}</div></div><p class="gate-note">${decisions.length?`the retrieval gate skipped memory on ${Math.round(skipped/total*100)}% of traces — latency and bias saved`:'send a trace and the retrieval gate starts deciding'}</p><section class="panel architecture"><div class="panel-head"><h2>Architecture — click any box <span class="arch-status"></span></h2><div class="legend"><span class="run">running</span><span class="ok">completed</span><span class="wait">waiting</span></div></div><div id="topology-wrap" class="topology-wrap"></div></section><section class="panel"><div class="panel-head"><h2>History replay</h2><div class="replay-controls"><button id="play" class="button">Play</button><button id="step" class="button">Step</button><select id="speed"><option value="800">1×</option><option value="400">2×</option></select></div></div><div id="timeline" class="timeline"></div></section><section class="panel"><div class="panel-head"><h2>Recent traces</h2></div>${turnTable(S.turns.slice(0,12))}</section>`;drawTopology();timeline();bindTurns();$('#play').onclick=togglePlay;$('#step').onclick=step}

function drawTopology(){if(!S.topology)return;const N=new Map(S.topology.nodes.map(n=>[n.id,n]));const edges=S.topology.edges.map(e=>{const a=N.get(e.source),b=N.get(e.target);if(!a||!b)return'';return `<line id="edge-${esc(e.id)}" class="edge ${e.optional?'optional':''}" x1="${a.x+a.w/2}" y1="${a.y+a.h/2}" x2="${b.x+b.w/2}" y2="${b.y+b.h/2}" marker-end="url(#arr)"/>`}).join('');const nodes=S.topology.nodes.map(n=>`<g id="node-${esc(n.id)}" data-node="${esc(n.id)}" class="node ${n.enabled?'':'disabled'}"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}"/><text class="label" x="${n.x+12}" y="${n.y+24}">${esc(n.label)}</text><text class="sub" x="${n.x+12}" y="${n.y+43}">${esc(n.sub)}</text></g>`).join('');const groups=`<text class="group-label" x="24" y="18">RUNTIME — ONE OBSERVABLE TRACE</text><rect class="group-boundary" x="12" y="28" width="1195" height="125" rx="14"/><text class="group-label" x="210" y="188">MEMORY &amp; RETRIEVAL — ON DEMAND</text><rect class="group-boundary memory" x="200" y="195" width="810" height="100" rx="12"/><text class="group-label" x="330" y="326">ACTION SURFACE</text><rect class="group-boundary actions" x="320" y="333" width="800" height="185" rx="12"/><text class="group-label" x="200" y="567">PERSISTENCE &amp; OPS — AFTER THE REPLY</text><rect class="group-boundary persistence" x="190" y="575" width="1035" height="118" rx="12"/>`;$('#topology-wrap').innerHTML=`<svg class="topology" viewBox="0 0 1240 720"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--ink3)"/></marker></defs>${groups}${edges}${nodes}</svg>`;document.querySelectorAll('[data-node]').forEach(e=>e.onclick=()=>openNode(e.dataset.node));applyVisual()}
function applyVisual(){document.querySelectorAll('.topology .node').forEach(e=>{e.classList.remove('active','success','waiting','error');const s=S.nodeStates.get(e.dataset.node);if(s)e.classList.add(s==='running'?'active':s)});document.querySelectorAll('.topology .edge').forEach(e=>e.classList.remove('active'));const ev=S.replay?.events[S.replay.index]||S.events.at(-1);(ev?.flow?.active_edges||[]).forEach(id=>document.querySelector(`#edge-${CSS.escape(id)}`)?.classList.add('active'))}
function applyEvent(e,live=false){if(e.event_id&&S.seen.has(e.event_id))return;if(e.event_id)S.seen.add(e.event_id);S.events.push(e);const f=e.flow||{};(f.active_nodes||[]).forEach(n=>S.nodeStates.set(n,f.state||'running'));$('#phase-label').textContent=f.phase||e.type;$('#stage-bar i').style.width=`${Math.min(100,Math.max(5,(e.sequence||1)*7))}%`;if(['trace.started','trace.accepted'].includes(e.type)){S.nodeStates.clear();setRuntime({busy:true,active_turn_id:e.trace_id||e.turn_id})}if(['trace.done','trace.aborted','trace.error','trace.failed'].includes(e.type)){setRuntime({busy:false,active_turn_id:''});refreshTurns()}if(e.type==='tool.approval.required')approval(e.data);if(e.type==='tool.approval.resolved')document.querySelector(`[data-approval="${CSS.escape(e.data.id)}"]`)?.remove();if(live)chatEvent(e);applyVisual();timeline()}

function timeline(){const el=$('#timeline');if(!el)return;const list=S.replay?.events||S.events.slice(-45);el.innerHTML=list.map((e,i)=>`<button data-i="${i}" class="${S.replay?.index===i?'active':''}">${esc(e.type)}</button>`).join('');el.querySelectorAll('button').forEach(b=>b.onclick=()=>{const i=+b.dataset.i;if(S.replay){S.replay.index=i;showReplay()}else openEvent(list[i])})}
function turnTable(rows){return `<table><thead><tr><th>Trace</th><th>source</th><th>status</th><th>time</th><th>message</th></tr></thead><tbody>${rows.map(t=>`<tr data-turn="${esc(t.trace_id||t.turn_id)}"><td><code>${esc((t.trace_id||t.turn_id).slice(0,8))}</code></td><td>${esc(t.source)}</td><td><span class="badge">${esc(t.status)}</span></td><td>${ms(t.duration_ms)}</td><td>${esc((t.message||'').slice(0,78))}</td></tr>`).join('')}</tbody></table>`}
function bindTurns(){document.querySelectorAll('[data-turn]').forEach(r=>r.onclick=()=>replay(r.dataset.turn))}
async function replay(id){const d=await api(`/api/turns/${encodeURIComponent(id)}/events`);S.replay={events:d.events,index:0,playing:false};showReplay();toast(`replay ${id.slice(0,8)}`)}
function showReplay(){const e=S.replay?.events[S.replay.index];if(!e)return;S.nodeStates.clear();(e.flow?.active_nodes||[]).forEach(n=>S.nodeStates.set(n,e.flow.state||'running'));applyVisual();timeline();openEvent(e)}
function step(){if(!S.replay){if(S.turns[0])replay(S.turns[0].turn_id);return}S.replay.index=Math.min(S.replay.events.length-1,S.replay.index+1);showReplay()}
function togglePlay(){if(!S.replay){if(S.turns[0])replay(S.turns[0].turn_id);return}S.replay.playing=!S.replay.playing;$('#play').textContent=S.replay.playing?'Pause':'Play';clearInterval(S.timer);if(S.replay.playing)S.timer=setInterval(()=>{if(S.replay.index>=S.replay.events.length-1){togglePlay();return}step()},+$('#speed').value)}

function loop(){$('#main').innerHTML=pageHead('Loop','Trace 列表、Turn 事件瀑布、重试、耗时和错误')+`<section class="panel">${turnTable(S.turns)}</section><section class="panel"><div class="timeline">${S.events.slice(-100).map((e,i)=>`<button data-last="${i}">${esc(e.type)} · ${ms(e.duration_ms)}</button>`).join('')}</div></section>`;bindTurns();document.querySelectorAll('[data-last]').forEach(b=>b.onclick=()=>openEvent(S.events.slice(-100)[+b.dataset.last]))}

function openNode(id){const n=S.topology.nodes.find(x=>x.id===id),events=S.events.filter(e=>(e.flow?.active_nodes||[]).includes(id)).slice(-20);openDrawer(`<h2>${esc(n?.label||id)}</h2><p>${esc(n?.sub||'')}</p><p><span class="badge">${esc(n?.health||'unknown')}</span></p><h3>Recent events</h3>${events.map(e=>`<div class="event-card"><b>${esc(e.type)}</b><br><small>${esc(e.timestamp)} · ${ms(e.duration_ms)}</small></div>`).join('')||'<p>暂无事件</p>'}`)}
function openEvent(e){if(e)openDrawer(`<h2>${esc(e.type)}</h2><p>${esc(e.timestamp)} · sequence ${esc(e.sequence)} · ${ms(e.duration_ms)}</p><pre>${esc(JSON.stringify(e,null,2))}</pre>`)}
function openDrawer(h){$('#drawer-content').innerHTML=h;$('#drawer').classList.add('open')}

async function dataPage(route){const endpoints={memory:'/api/memory',rag:'/api/rag',tools:'/api/tools',subagents:'/api/subagents',files:'/api/files',database:'/api/database',ops:'/api/ops',settings:'/api/settings'};try{const d=await api(endpoints[route]);RENDER[route](d)}catch(e){$('#main').innerHTML=pageHead(route,e.message)}}
const cards=(a,fn)=>`<div class="grid">${a.length?a.map(fn).join(''):'<article class="data-card"><p>暂无数据</p></article>'}</div>`;
const RENDER={
 memory(d){$('#main').innerHTML=pageHead('Memory','Semantic · Episodic · Procedural · Consolidation')+`<div class="metric-row">${metric('facts',d.facts.length)}${metric('episodes',d.episodes.length)}${metric('summary',d.summary?`v${d.summary.version}`:'none')}${metric('gate','observable')}</div>${cards([{title:'Rolling summary',text:d.summary?.summary||'暂无摘要'},...d.facts.map(x=>({title:`Fact · ${x.subject}`,text:x.content})),...d.episodes.map(x=>({title:`Episode · ${x.happened_at}`,text:x.summary}))],x=>`<article class="data-card"><h3>${esc(x.title)}</h3><p>${esc(x.text)}</p></article>`)}`},
 rag(d){$('#main').innerHTML=pageHead('RAG','文档、chunk、Token、召回与 rerank')+`<div class="metric-row">${metric('enabled',d.enabled?'yes':'no')}${metric('documents',d.stats.documents||0)}${metric('chunks',d.stats.chunks||0)}${metric('tokens',d.stats.total_tokens||0)}</div>${cards(d.documents,x=>`<article class="data-card"><h3>${esc(x.title)}</h3><p>${esc(x.path)}</p><span class="badge">${esc(x.chunk_count)} chunks</span></article>`)}`},
 tools(d){$('#main').innerHTML=pageHead('Tools','Native · MCP · Sandbox · approval')+`<div class="metric-row">${metric('available',d.tools.length)}${metric('pending approval',d.approvals.pending.length)}${metric('approval history',d.approvals.history.length)}${metric('host shell',d.tools.some(x=>x.name==='exec')?'visible':'hidden')}</div>${cards(d.tools,x=>`<article class="data-card"><h3><code>${esc(x.name)}</code> <span class="badge ${esc(x.effect)}">${esc(x.effect)}</span></h3><p>${esc(x.description)}</p><small>${x.sandboxed?'Docker sandbox':'internal / host runtime'}</small></article>`)}`},
 subagents(d){$('#main').innerHTML=pageHead('Subagents','独立 Session、并行状态、工具轨迹与结果')+cards(d.subagents,x=>`<article class="data-card"><h3>${esc(x.label)} <span class="badge">${esc(x.phase)}</span></h3><p>${esc(x.task)}</p><small>${esc(x.tools_called.join(', '))} · ${esc(x.elapsed)}s</small><p>${esc(x.reply||x.error||'running')}</p></article>`)},
 files(d){$('#main').innerHTML=pageHead('Files','每轮变更、diff 与撤销状态')+`<div class="metric-row">${metric('files',d.modified_files.length)}${metric('last turn',d.last_summary)}${metric('undo',d.modified_files.length?'available':'empty')}${metric('scope','workspace')}</div><section class="panel"><pre>${esc(d.diff||'暂无进行中的文件变更')}</pre></section>`},
 database(d){$('#main').innerHTML=pageHead('Database','只读表结构与行数；不提供任意 SQL')+`<section class="panel"><table><thead><tr><th>table</th><th>rows</th></tr></thead><tbody>${d.tables.map(x=>`<tr><td><code>${esc(x.name)}</code></td><td>${esc(x.rows)}</td></tr>`).join('')}</tbody></table></section>`},
 ops(d){$('#main').innerHTML=pageHead('Ops','Trace · Usage · Doctor · Eval')+`<div class="metric-row">${metric('input tokens',d.usage.total_input)}${metric('output tokens',d.usage.total_output)}${metric('recent events',d.events.length)}${metric('runtime',d.runtime.busy?'busy':'idle')}</div><section class="panel"><pre>${esc(d.events.slice(-40).map(e=>`${e.timestamp}  ${e.type}  ${JSON.stringify(e.data)}`).join('\n'))}</pre></section>`},
 settings(d){$('#main').innerHTML=pageHead('Settings','Model 与 thinking；API Key 永不进入页面')+`<div class="grid"><article class="data-card"><h3>Model</h3><select id="provider">${d.providers.map(p=>`<option value="${esc(p.id)}" ${p.id===d.provider?'selected':''}>${esc(p.id)} · ${esc(p.model)}${p.configured?' · configured':''}</option>`).join('')}</select> <button id="model-save" class="button">Switch</button></article><article class="data-card"><h3>Thinking</h3><select id="thinking"><option>disabled</option><option>auto</option><option>enabled</option></select> <button id="thinking-save" class="button">Save</button></article><article class="data-card"><h3>Web security</h3><p>localhost only · random per-start token · strict CSP</p><p>host shell: ${d.web_allow_host_shell?'explicitly allowed':'hidden'}</p></article><article class="data-card"><h3>Capabilities</h3><p>Sandbox ${d.sandbox?'on':'off'} · RAG ${d.rag?'on':'off'} · MCP ${d.mcp?'on':'off'}</p></article></div>`;$('#thinking').value=d.thinking;$('#model-save').onclick=async()=>{await api('/api/settings/model',{method:'POST',body:JSON.stringify({provider:$('#provider').value})});toast('model switched');boot()};$('#thinking-save').onclick=async()=>{await api('/api/settings/thinking',{method:'POST',body:JSON.stringify({thinking:$('#thinking').value})});toast('thinking updated');boot()}}
};

function clearCarets(root=document){root.querySelectorAll('.stream-caret').forEach(node=>node.remove())}
async function send(message){if(S.busy)return;const log=$('#chat-log');log.querySelector('.empty-chat')?.remove();clearCarets(log);document.querySelector('#stream-reply')?.removeAttribute('id');log.insertAdjacentHTML('beforeend',`<div class="message user">${esc(message)}</div><div class="message assistant" id="stream-reply"><span class="stream-caret"></span></div>`);setRuntime({busy:true,active_turn_id:''});try{const r=await fetch('/api/turns/stream',{method:'POST',headers:{'X-LSM-Web-Token':TOKEN,'Content-Type':'application/json'},body:JSON.stringify({message})});if(!r.ok)throw new Error((await r.json()).error||r.statusText);const reader=r.body.getReader(),dec=new TextDecoder();let buf='';for(;;){const {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;while((i=buf.indexOf('\n\n'))>=0){const packet=buf.slice(0,i);buf=buf.slice(i+2);const line=packet.split('\n').find(x=>x.startsWith('data: '));if(!line)continue;applyEvent(JSON.parse(line.slice(6)),true)}}}catch(e){clearCarets(log);toast(e.message)}finally{clearCarets(log);setRuntime({busy:false,active_turn_id:''});refreshTurns()}}
function chatEvent(e){
  if(e.type==='transport.ready'){S.activeTurn=e.turn_id;return}
  const reply=$('#stream-reply');
  const beforeReply=html=>reply?reply.insertAdjacentHTML('beforebegin',html):$('#chat-log').insertAdjacentHTML('beforeend',html);
  if(e.type==='llm.text.delta'&&reply){
    reply.classList.remove('rendered','awaiting-reply');
    clearCarets(reply);
    reply.append(document.createTextNode(e.data.text||''));
    reply.insertAdjacentHTML('beforeend','<span class="stream-caret"></span>');
  }else if(e.type==='tool.requested'){
    // Text emitted before a tool call is an intermediate model turn, not the
    // user-facing answer. Waku clears the same buffer at this boundary.
    if(reply){reply.replaceChildren();reply.classList.add('awaiting-reply')}
    clearCarets(document);
    beforeReply(`<div class="event-card"><b>⚙ ${esc(e.data.tool)}</b><br><small>requested</small></div>`);
  }else if(e.type==='tool.completed'){
    beforeReply(`<div class="event-card"><b>${e.data.status==='error'?'✕':'✓'} ${esc(e.data.tool)}</b><br><small>${esc((e.data.output||'').slice(0,140))}</small></div>`);
  }else if(e.type==='subagent.started'){
    beforeReply(`<div class="event-card"><b>↗ ${esc(e.data.label)}</b><br><small>subagent running</small></div>`);
  }
  if(['llm.text.end','llm.completed','trace.done','trace.completed','trace.aborted','trace.error','trace.failed','transport.error'].includes(e.type)){
    clearCarets(reply||document);
    if(['trace.done','trace.completed'].includes(e.type)&&reply&&e.data?.reply){
      reply.innerHTML=renderMarkdown(e.data.reply);
      reply.classList.remove('awaiting-reply');
      reply.classList.add('rendered');
    }
  }
  $('#chat-log').scrollTop=$('#chat-log').scrollHeight;
}
function approval(d){$('#approval-zone').innerHTML=`<div class="approval-card" data-approval="${esc(d.id)}"><b>${esc(d.tool_name)}</b> · ${esc(d.effect)}<p>${esc(JSON.stringify(d.arguments))}</p><small>${d.sandboxed?'Docker sandbox':'not sandboxed'} · 120s timeout</small><div class="actions"><button class="button primary" data-decision="approve">Allow</button><button class="button danger" data-decision="reject">Reject</button></div></div>`;document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=async()=>{await api(`/api/approvals/${d.id}`,{method:'POST',body:JSON.stringify({decision:b.dataset.decision})});b.closest('.approval-card').remove()})}

async function poll(){try{const d=await api(`/api/events?cursor=${S.cursor}`);S.cursor=d.cursor;d.events.forEach(e=>applyEvent(e));if(S.route==='overview'&&d.events.length)overview()}catch{}setTimeout(poll,1000)}
async function refreshTurns(){try{S.turns=(await api('/api/turns')).turns;if(['overview','loop'].includes(S.route))render()}catch{}}
document.querySelectorAll('#nav button').forEach(b=>b.onclick=()=>{S.route=b.dataset.route;location.hash=S.route;render()});
$('#drawer-close').onclick=()=>$('#drawer').classList.remove('open');
$('#chat-form').onsubmit=e=>{e.preventDefault();const input=$('#message-input'),m=input.value.trim();if(m){input.value='';send(m)}};
$('#stop-turn').onclick=async()=>{if(S.activeTurn)await api(`/api/turns/${S.activeTurn}/abort`,{method:'POST',body:'{}'})};
$('#steer-send').onclick=async()=>{const m=$('#steer-input').value.trim();if(m&&S.activeTurn){await api(`/api/turns/${S.activeTurn}/steer`,{method:'POST',body:JSON.stringify({message:m})});$('#steer-input').value=''}};
$('#new-chat').onclick=async()=>{await api('/api/sessions',{method:'POST',body:'{}'});$('#chat-log').innerHTML='<div class="empty-chat">新会话已创建。</div>';boot()};
$('#history-button').onclick=()=>{S.route='loop';render()};
function wireResizer(id,variable,key,right,min,max){const el=document.getElementById(id);if(!el)return;el.onmousedown=e=>{e.preventDefault();document.body.classList.add('resizing');const move=ev=>{let w=right?innerWidth-ev.clientX:ev.clientX;w=Math.max(min,Math.min(max,w));document.documentElement.style.setProperty(variable,w+'px');localStorage.setItem(key,w)};const up=()=>{document.body.classList.remove('resizing');removeEventListener('mousemove',move);removeEventListener('mouseup',up)};addEventListener('mousemove',move);addEventListener('mouseup',up)}}
const nw=localStorage.getItem('lsmNavW'),dw=localStorage.getItem('lsmDockW');if(nw)document.documentElement.style.setProperty('--nav-w',nw+'px');if(dw)document.documentElement.style.setProperty('--dock-w',dw+'px');wireResizer('nav-resizer','--nav-w','lsmNavW',false,150,360);wireResizer('dock-resizer','--dock-w','lsmDockW',true,280,650);
const setNav=v=>{document.body.classList.toggle('nav-hidden',v);localStorage.setItem('lsmNavHidden',v?'1':'0')};$('#nav-toggle').onclick=()=>setNav(true);$('#nav-reopen').onclick=()=>setNav(false);setNav(localStorage.getItem('lsmNavHidden')==='1');
const setDock=v=>{document.body.classList.toggle('dock-closed',v);localStorage.setItem('lsmDockClosed',v?'1':'0')};$('#dock-close').onclick=()=>setDock(true);$('#dock-reopen').onclick=()=>setDock(false);const savedDock=localStorage.getItem('lsmDockClosed');setDock(savedDock==='1'||(savedDock===null&&innerWidth<1000));
boot().catch(e=>{$('#main').innerHTML=pageHead('LSM Console unavailable',e.message)});
