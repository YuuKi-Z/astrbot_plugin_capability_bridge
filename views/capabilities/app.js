const $ = (id) => document.getElementById(id);
const bridge = window.AstrBotPluginView;
let snapshot;
let settingsLoaded = false;
let busy = false;
let serialized = "";
const labels = {ready:'已适配',native:'原生工具',passive:'自动行为',new:'待分析',stale:'需要重新适配',disabled:'插件停用',unsupported:'暂无可用适配',queued:'等待中',analyzing:'分析中',completed:'已完成',failed:'失败',cancelled:'已停止'};
function node(tag, text, className) { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(className)e.className=className; return e; }
function notice(text, error=false) { $('notice').hidden=false; $('notice').textContent=text; $('notice').className=error?'error':''; }
async function action(endpoint, body={}, message='已完成') {
  if(busy)return;
  busy=true;
  try { await bridge.apiPost(endpoint,body); notice(message); await refresh(); }
  catch(e) {notice(e.message||String(e),true);}
  finally {busy=false;}
}
function render() {
  const s=snapshot;
  const opened=new Set([...document.querySelectorAll('article[data-plugin]')].filter(e=>e.querySelector('details')?.open).map(e=>e.dataset.plugin));
  $('connection').textContent=s.enabled?'已连接 · 能力桥运行中':'能力桥已停用';
  if(!settingsLoaded){
    $('provider').replaceChildren(node('option','选择已配置的聊天模型'));
    $('provider').firstChild.value='';
    for(const p of s.providers){const opt=node('option',`${p.id} · ${p.model}`);opt.value=p.id;$('provider').append(opt);}
    $('provider').value=s.settings.provider_id;
    settingsLoaded=true;
  }
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
  $('usage').textContent=`实际用量 ${(s.job.tokens||0).toLocaleString()} tokens${s.job.estimated_tokens?`；未返回用量的请求估计 ${s.job.estimated_tokens.toLocaleString()}`:''}`;
  $('job-items').replaceChildren(...items.map(i=>node('li',`${i.plugin} · ${labels[i.status]||i.status}${i.capabilities!==undefined?` · ${i.capabilities} 项能力`:''}${i.error?` · ${i.error}`:''}`)));
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
      if(p.native_tools.length)detail.append(node('p',`可直接调用的原生工具：${p.native_tools.join('、')}`,'small'));
      for(const cap of p.capabilities){
        const row=node('div',undefined,'cap'), info=node('div',undefined,'cap-info');info.append(node('h4',cap.name),node('p',cap.description));
        if(cap.examples.length)info.append(node('p',`试着说：${cap.examples[0]}`,'example'));
        const toggle=node('button',cap.enabled?'已启用 · 点击停用':'已停用 · 点击启用');toggle.onclick=()=>action('toggle',{id:cap.id,enabled:!cap.enabled},'能力开关已保存');row.append(info,toggle);detail.append(row);
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
$('save').onclick=()=>action('settings',{provider_id:$('provider').value},'设置已保存');
$('adapt').onclick=()=>action('adapt',{},'适配任务已启动，可以在下方查看进度');
$('cancel').onclick=()=>action('cancel',{},'已请求停止，已完成的适配会保留');
$('search').oninput=()=>{if(snapshot)render();};
try {if(!bridge)throw Error('请在 AstrBot 插件详情的可视化视图中打开能力中心');await bridge.ready();await refresh();setInterval(()=>{if(!busy)refresh().catch(e=>notice(e.message||String(e),true));},3000);}
catch(e){$('connection').textContent='连接失败';notice(e.message||String(e),true);}
