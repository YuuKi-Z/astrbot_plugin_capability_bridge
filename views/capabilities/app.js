const $ = (id) => document.getElementById(id);
const bridge = window.AstrBotPluginView;
let snapshot;
let analysisDirty = false;
let settingsDirty = false;
let analysisRevision = '';
let settingsRevision = '';
let busy = false;
let serialized = "";
const labels = {ready:'已适配',native:'原生工具',passive:'自动行为',new:'待分析',stale:'需要重新适配',disabled:'插件停用',unsupported:'暂无可用适配',queued:'等待中',analyzing:'分析中',completed:'已完成',failed:'失败',cancelled:'已停止'};
function node(tag, text, className) { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(className)e.className=className; return e; }
function notice(text, error=false) { $('notice').hidden=false; $('notice').textContent=text; $('notice').className=error?'error':''; }
async function action(endpoint, body={}, message='已完成') {
  if(busy)return false;
  busy=true;
  try { await bridge.apiPost(endpoint,body); notice(message); await refresh(); return true; }
  catch(e) {notice(e.message||String(e),true);return false;}
  finally {busy=false;}
}
function providers(select,value,emptyLabel) {
  const list=snapshot.providers, key=JSON.stringify([list,value&& !list.some(p=>p.id===value)?value:null]);
  if(select.dataset.options!==key){
    const empty=node('option',emptyLabel);empty.value='';select.replaceChildren(empty);
    for(const p of list){const opt=node('option',`${p.id} · ${p.model}`);opt.value=p.id;select.append(opt);}
    if(value&&!list.some(p=>p.id===value)){const missing=node('option',`${value} · 当前不可用`);missing.value=value;missing.disabled=true;select.append(missing);}
    select.dataset.options=key;
  }
  select.value=value||'';
}
function showPage(name) {
  for(const page of ['analysis','settings']){const active=page===name;$(`page-${page}`).hidden=!active;$(`tab-${page}`).setAttribute('aria-selected',String(active));}
}
function parseList(id){return $(id).value.split(/[\n,，]+/).map(v=>v.trim()).filter(Boolean);}
function render() {
  const s=snapshot;
  const confirmation=s.settings.confirmation_enabled!==false;
  const policyLabels=confirmation?{confirm:'需本人确认',direct:'直接调用',admin_confirm:'仅管理员且需确认'}:{confirm:'直接调用（确认已关闭）',direct:'直接调用',admin_confirm:'仅管理员（确认已关闭）'};
  const opened=new Set([...document.querySelectorAll('article[data-plugin]')].filter(e=>e.querySelector('details')?.open).map(e=>e.dataset.plugin));
  $('connection').textContent=s.enabled?'已连接 · 自然语言指令运行中':'自然语言指令已停用';
  if(!analysisDirty){
    providers($('provider'),s.settings.analysis_provider_id,'请选择分析模型');
    $('analysis-timeout').value=s.settings.analysis_timeout||600;
    analysisRevision=s.settings_revision;
  }
  if(!settingsDirty){
    providers($('execution-provider'),s.settings.execution_provider_id,'沿用 AstrBot 当前模型');
    $('setting-enabled').checked=s.settings.enabled;
    $('setting-confirmation').checked=confirmation;
    $('setting-private').checked=s.settings.allow_private_chat;
    $('setting-whitelist-enabled').checked=s.settings.group_whitelist_enabled;
    $('group-whitelist').value=(s.settings.group_whitelist||[]).join('\n');
    $('group-blacklist').value=(s.settings.group_blacklist||[]).join('\n');
    $('user-blacklist').value=(s.settings.user_blacklist||[]).join('\n');
    settingsRevision=s.settings_revision;
  }
  $('policy-summary').textContent=(confirmation?'二次确认已开启；新能力默认需本人确认。':'二次确认已关闭；“需确认”能力将直接调用，“仅管理员”限制仍然有效。')+' 原插件权限始终有效，插件更新后请重新审阅。调用方式适用于适配指令，原生工具沿用 AstrBot 与原插件的控制。';
  $('count-ready').textContent=s.plugins.filter(p=>p.state==='ready').reduce((n,p)=>n+p.capabilities.filter(c=>c.enabled).length,0);
  $('count-native').textContent=s.plugins.reduce((n,p)=>n+(p.active?p.native_tools.length:0),0);
  $('count-changed').textContent=s.plugins.filter(p=>p.active&&p.command_count&&['new','stale'].includes(p.state)).length;
  const running=s.job.status==='running';
  $('adapt').disabled=running; $('save').disabled=running;
  $('job').hidden=s.job.status==='idle';
  $('job-title').textContent=running?'正在适配插件':s.job.status==='cancelled'?'适配已停止':'本轮适配结果';
  $('cancel').hidden=!running;
  const items=s.job.items||[];
  $('progress').max=Math.max(1,items.length);$('progress').value=items.filter(i=>!['queued','analyzing'].includes(i.status)).length;
  $('usage').textContent=`服务商已报告 ${(s.job.tokens||0).toLocaleString()} tokens${s.job.estimated_tokens?`；已完成但未报告用量的响应约 ${s.job.estimated_tokens.toLocaleString()} tokens`:''}${s.job.pending_requests?'；1 个请求进行中，用量待返回':''}${s.job.unknown_requests?`；${s.job.unknown_requests} 个中断请求用量未知`:''}。以服务商账单为准。`;
  const phases={waiting:'等待模型响应',thinking:'模型正在思考',receiving:'正在接收适配结果',validating:'正在校验'};
  $('job-items').replaceChildren(...items.map(i=>node('li',`${i.plugin} · ${labels[i.status]||i.status}${i.batch?` · 第 ${i.batch} 批`:''}${i.status==='analyzing'?` · ${phases[i.phase]||'准备请求'} · 本批 ${i.elapsed_seconds||0}/${i.timeout_seconds||600} 秒${i.received_chars?` · 已收到 ${i.received_chars.toLocaleString()} 字符`:''}`:''}${i.resumed_batches?` · 复用 ${i.resumed_batches} 个已完成批次`:''}${i.capabilities!==undefined?` · ${i.capabilities} 项能力`:''}${i.error?` · ${i.error}`:''}`)));
  const search=$('search').value.toLowerCase();
  $('plugins').replaceChildren();
  for(const p of s.plugins){
    if(search&&!`${p.name} ${p.display_name} ${p.capabilities.map(c=>c.name).join(' ')}`.toLowerCase().includes(search))continue;
    const box=node('article',undefined,'plugin');box.dataset.plugin=p.name;const head=node('div',undefined,'plugin-head');const name=node('div',undefined,'name');
    name.append(node('h3',p.display_name),node('small',`${p.name} · ${p.version||'未标版本'} · ${p.command_count} 条指令 / ${p.native_tools.length} 个原生工具`));
    const btn=node('button',p.state==='stale'?'重新适配':'分析适配');btn.disabled=running||!p.active||!p.command_count;btn.onclick=()=>action('adapt',{plugins:[p.name]},'适配任务已启动');
    head.append(name,node('span',labels[p.state],`tag ${p.state}`),btn);box.append(head);
    if(p.capabilities.length||p.native_tools.length||p.unsupported.length||p.error){
      const detail=node('details',undefined,'plugin-body');detail.open=opened.has(p.name);detail.append(node('summary',`查看能力与适配详情${p.tokens?` · 上次 ${p.tokens.toLocaleString()} tokens`:''}`));
      if(p.state==='stale')detail.append(node('p','插件源码已更新，以下旧规则暂停使用。重新适配后恢复。','small'));
      if(p.native_tools.length)detail.append(node('p',`原生工具（沿用 AstrBot 与原插件的调用控制）：${p.native_tools.join('、')}`,'small'));
      for(const cap of p.capabilities){
        const row=node('div',undefined,'cap'), info=node('div',undefined,'cap-info');info.append(node('h4',cap.name),node('p',cap.description));
        if(cap.examples.length)info.append(node('p',`试着说：${cap.examples[0]}`,'example'));
        if(!cap.policy_reviewed)info.append(node('p',cap.call_policy==='admin_confirm'?'新版调用方式尚未审阅，保留仅管理员限制。':confirmation?'尚未审阅调用方式，默认需本人确认。':'尚未审阅调用方式；二次确认已关闭，当前直接调用。','small'));
        const controls=node('div',undefined,'cap-actions'),policy=node('select');policy.setAttribute('aria-label',`${cap.name}的调用方式`);
        for(const [value,label] of Object.entries(policyLabels)){const option=node('option',label);option.value=value;policy.append(option);}
        policy.value=cap.call_policy||'confirm';policy.disabled=running||p.state==='stale';
        policy.onchange=async()=>{await action('policy',{id:cap.id,mode:policy.value,fingerprint:p.fingerprint},'调用方式已保存，原插件权限仍然有效');render();};
        const toggle=node('button',cap.enabled?'已启用 · 点击停用':'已停用 · 点击启用');toggle.onclick=()=>action('toggle',{id:cap.id,enabled:!cap.enabled},'能力开关已保存');controls.append(policy,toggle);row.append(info,controls);detail.append(row);
      }
      if(p.unsupported.length){const skipped=node('details');skipped.append(node('summary',`${p.unsupported.length} 项未适配或校验未通过`));for(const u of p.unsupported)skipped.append(node('p',`${u.command_id.split('.').pop()}：${u.reason}`,'unsupported'));detail.append(skipped);}
      if(p.error)detail.append(node('p',p.error,'unsupported'));box.append(detail);
    }
    $('plugins').append(box);
  }
  if(!$('plugins').children.length)$('plugins').append(node('p','没有匹配的插件。','empty'));
}
async function refresh(){const next=await bridge.apiGet('status');const key=JSON.stringify(next);if(key===serialized)return;serialized=key;snapshot=next;render();}
$('scan').onclick=()=>action('scan',{},'扫描完成，未调用分析模型');
$('provider').onchange=()=>{analysisDirty=true;};
$('analysis-timeout').oninput=()=>{analysisDirty=true;};
$('save').onclick=async()=>{if(!$('analysis-timeout').reportValidity())return;if(await action('settings',{analysis_provider_id:$('provider').value,analysis_timeout:Number($('analysis-timeout').value),revision:analysisRevision},'分析设置已保存')){analysisDirty=false;render();}};
$('save-settings').onclick=async()=>{
  const data={enabled:$('setting-enabled').checked,confirmation_enabled:$('setting-confirmation').checked,execution_provider_id:$('execution-provider').value,allow_private_chat:$('setting-private').checked,group_whitelist_enabled:$('setting-whitelist-enabled').checked,group_whitelist:parseList('group-whitelist'),group_blacklist:parseList('group-blacklist'),user_blacklist:parseList('user-blacklist'),revision:settingsRevision};
  if(await action('settings',data,'设置已保存；运行设置变化时，旧确认码已失效')){settingsDirty=false;render();}
};
for(const id of ['execution-provider','setting-enabled','setting-confirmation','setting-private','setting-whitelist-enabled','group-whitelist','group-blacklist','user-blacklist']){$(id).addEventListener('input',()=>{settingsDirty=true;});$(id).addEventListener('change',()=>{settingsDirty=true;});}
$('reload-settings').onclick=async()=>{settingsDirty=false;try{await refresh();render();notice('已重新载入设置');}catch(e){notice(e.message||String(e),true);}};
for(const name of ['analysis','settings']){$(`tab-${name}`).onclick=()=>{location.hash=name;showPage(name);};}
window.addEventListener('hashchange',()=>showPage(location.hash==='#settings'?'settings':'analysis'));
showPage(location.hash==='#settings'?'settings':'analysis');
$('adapt').onclick=()=>action('adapt',{},'适配任务已启动，可以在下方查看进度');
$('cancel').onclick=()=>action('cancel',{},'已请求停止，已完成的适配会保留');
$('search').oninput=()=>{if(snapshot)render();};
try {if(!bridge)throw Error('请在 AstrBot 插件详情的可视化视图中打开能力中心');await bridge.ready();await refresh();setInterval(()=>{if(!busy)refresh().catch(e=>notice(e.message||String(e),true));},3000);}
catch(e){$('connection').textContent='连接失败';notice(e.message||String(e),true);}
