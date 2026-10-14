'use client';
import {useCallback,useEffect,useState} from 'react';
import {AlertTriangle,CheckCircle2,CircleHelp,FileText,Layers,Play,Printer,ShieldAlert,Upload} from 'lucide-react';
import {getJSON} from '../lib/api';
import {Protocol} from './external-predictions';

type Series={id:string;plane:string;format:string;metadata:Record<string,unknown>};
type Finding={finding:string;probability:number;seed_min:number;seed_max:number;review_required:boolean;reasons:string[]};
type Reference={finding:string;auc:number|null;brier:number;expert_positives:number;expert_n:number};
type Result={findings:Finding[];final_model_id:string;reference_performance:{macro_auc:number;per_finding:Reference[];source:string}|null;
 feature_provenance:{sources:{slot:number;series_id:string}[];errors:{series_id:string;error:string}[]};policy:{lower:number;upper:number}};
type Job={id:string;status:string;progress:number;message:string;error:string|null;result:Result|Record<string,never>;created_at:string};
type Case={id:string;split:string;source:string;original_name:string|null;series:Series[];latest_job:Job|null};
type ModelInfo={available:boolean;reason?:string;id?:string;training_studies?:number;expert_target_studies?:number;seeds?:number[];
 review_policy?:{lower:number;upper:number;max_seed_range:number};reference_performance?:{macro_auc:number}|null};

const GROUPS:[string,string[]][]=[['Ligaments',['ACL','MCL']],['Menisci',['Medial Meniscus','Lateral Meniscus']],
 ['Cartilage / osteoarthritis',['Medial OA','Lateral OA','PF OA']],['Joint fluid & synovium',['Effusion','Synovitis',"Baker's"]],['Bone',['Contusion','Fracture']]];
const NAMES:Record<string,string>={'ACL':'Anterior cruciate ligament tear','MCL':'Medial collateral ligament injury','Medial Meniscus':'Medial meniscus tear',
 'Lateral Meniscus':'Lateral meniscus tear','Medial OA':'Medial compartment OA','Lateral OA':'Lateral compartment OA','PF OA':'Patellofemoral OA',
 'Effusion':'Joint effusion','Synovitis':'Synovitis',"Baker's":"Baker's (popliteal) cyst",'Contusion':'Bone contusion','Fracture':'Fracture'};
const short=(id:string)=>id.length>18?'…'+id.slice(-12):id;

function verdict(f:Finding,lower:number,upper:number){
 if(f.review_required&&f.probability>lower&&f.probability<upper)return {key:'review',label:'Indeterminate · radiologist review',icon:CircleHelp};
 if(f.probability>=upper)return {key:'present',label:f.review_required?'Likely present · verify':'Likely present',icon:AlertTriangle};
 if(f.probability<=lower)return {key:'absent',label:f.review_required?'Likely absent · verify':'Likely absent',icon:CheckCircle2};
 return {key:'review',label:'Indeterminate · radiologist review',icon:CircleHelp};
}
function reliability(auc:number|null|undefined){
 if(auc==null)return {label:'Not measurable',cls:''};
 if(auc>=0.8)return {label:`Good (AUC ${auc.toFixed(2)})`,cls:'good'};
 if(auc>=0.7)return {label:`Moderate (AUC ${auc.toFixed(2)})`,cls:'moderate'};
 return {label:`Weak (AUC ${auc.toFixed(2)})`,cls:'weak'};
}

function Viewer({series}:{series:Series[]}){
 const [current,setCurrent]=useState(series[0]?.id||''),[volume,setVolume]=useState<{shape:number[]}|null>(null),[slice,setSlice]=useState(0),[error,setError]=useState('');
 const plane=series.find(s=>s.id===current)?.plane;const axis=plane==='Sagittal'?0:plane==='Coronal'?1:2;
 useEffect(()=>{if(!current)return;let live=true;setVolume(null);setError('');getJSON<{shape:number[]}>(`/series/${encodeURIComponent(current)}/volume`)
  .then(v=>{if(live){setVolume(v);setSlice(Math.floor(v.shape[axis]/2))}}).catch(e=>{if(live)setError((e as Error).message)});return()=>{live=false}},[current,axis]);
 return <section className="panel viewer clinical-viewer"><div className="viewer-toolbar"><select aria-label="MRI sequence" value={current} onChange={e=>setCurrent(e.target.value)}>
  {series.map((s,i)=><option key={s.id} value={s.id}>{s.plane} · sequence {i+1}{s.metadata.fluid_sensitive_csv==1||s.metadata.fluid_sensitive==1?' · fluid-sensitive':''}</option>)}</select></div>
  <div className="image-area">{error?<div className="empty"><p>{error}</p></div>:volume?<img alt={`MRI slice ${slice+1}`} src={`/api/series/${encodeURIComponent(current)}/slice?axis=${axis}&index=${slice}`}/>:<div className="empty"><Layers size={30}/><p>Loading images…</p></div>}</div>
  <div className="slice-controls"><span>Slice {slice+1} / {volume?.shape[axis]||'—'}</span><input aria-label="Slice" type="range" min={0} max={(volume?.shape[axis]||1)-1} value={slice} onChange={e=>setSlice(Number(e.target.value))}/></div></section>;
}

function Findings({result}:{result:Result}){
 const lower=result.policy?.lower??0.2,upper=result.policy?.upper??0.8;
 const byName=Object.fromEntries(result.findings.map(f=>[f.finding,f]));
 const refs=Object.fromEntries((result.reference_performance?.per_finding||[]).map(r=>[r.finding,r]));
 const counts={present:0,absent:0,review:0} as Record<string,number>;result.findings.forEach(f=>counts[verdict(f,lower,upper).key]++);
 return <>
  <div className="clinical-summary">
   <div className="summary-tile present"><strong>{counts.present}</strong><span>Likely present</span></div>
   <div className="summary-tile review"><strong>{counts.review}</strong><span>Need radiologist review</span></div>
   <div className="summary-tile absent"><strong>{counts.absent}</strong><span>Likely absent</span></div>
  </div>
  {result.feature_provenance?.errors?.length>0&&<div className="alert error">Some sequences could not be processed: {result.feature_provenance.errors.map(e=>e.error).join('; ')}</div>}
  {GROUPS.map(([group,names])=><section className="panel finding-group" key={group}><h2>{group}</h2>
   {names.map(name=>{const f=byName[name];if(!f)return null;const v=verdict(f,lower,upper),Icon=v.icon,r=reliability(refs[name]?.auc);
    return <div className={'clinical-finding '+v.key} key={name}>
     <div className="cf-name"><Icon size={17}/><div><strong>{NAMES[name]||name}</strong><small>{v.label}</small></div></div>
     <div className="cf-bar" aria-label={`${name} probability ${(f.probability*100).toFixed(0)} percent`}>
      <div className="band" style={{left:lower*100+'%',width:(upper-lower)*100+'%'}}/>
      <div className="range" style={{left:f.seed_min*100+'%',width:Math.max(0.6,(f.seed_max-f.seed_min)*100)+'%'}}/>
      <div className="dot" style={{left:f.probability*100+'%'}}/>
     </div>
     <div className="cf-value">{(f.probability*100).toFixed(0)}%<small>{(f.seed_min*100).toFixed(0)}–{(f.seed_max*100).toFixed(0)}%</small></div>
     <div className={'cf-reliability '+r.cls} title="Cross-validated performance of this method for this finding on 58 expert-labelled studies">{r.label}</div>
    </div>})}
  </section>)}
  <p className="panel-note clinical-legend">Dot = model probability. Thin bar = spread across the 3 model seeds (training variability, not a confidence interval). Shaded zone = {lower*100}–{upper*100}% indeterminate band where the system abstains.
   Reliability = cross-validated AUC of this method for that finding on 58 expert-labelled studies{result.reference_performance?` (macro AUC ${result.reference_performance.macro_auc.toFixed(2)})`:''}. The deployed model was refit on all studies and has no independent test result.</p>
 </>;
}

export function ClinicalWorkspace(){
 const [cases,setCases]=useState<Case[]>([]),[model,setModel]=useState<ModelInfo|null>(null),[selected,setSelected]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[clinician,setClinician]=useState('');
 const load=useCallback(async()=>{try{const [c,m]=await Promise.all([getJSON<Case[]>('/clinical/studies'),getJSON<ModelInfo>('/final-model')]);setCases(c);setModel(m);setSelected(s=>s||c[0]?.id||'')}catch(e){setError((e as Error).message)}},[]);
 useEffect(()=>{load();const t=setInterval(load,3000);return()=>clearInterval(t)},[load]);
 const current=cases.find(c=>c.id===selected),job=current?.latest_job,running=!!job&&['queued','running'].includes(job.status);
 const result=job?.status==='completed'&&'findings' in (job.result||{})?job.result as Result:null;
 const needsProtocol=current?.split==='external'&&current.series.some(s=>!s.metadata.protocol_reviewed);
 async function analyze(){if(!current)return;setBusy(true);setError('');try{await getJSON('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'infer',model:'final',study_id:current.id})});await load()}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 async function upload(file:File){setBusy(true);setError('');try{const form=new FormData();form.append('file',file);const r=await getJSON<{study_ids:string[]}>('/uploads',{method:'POST',body:form});await load();if(r.study_ids[0])setSelected(r.study_ids[0])}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 return <div className="clinical">
  <div className="alert error clinical-disclaimer" role="note"><ShieldAlert size={18}/><span><strong>Decision support for research only — not a medical device.</strong> Not clinically validated. Every output must be confirmed by a qualified radiologist; never use it alone for diagnosis or treatment.</span></div>
  {error&&<div className="alert error" role="alert">{error}<button onClick={()=>setError('')}>Dismiss</button></div>}
  {model&&!model.available&&<div className="alert error">No deployed model: {model.reason}</div>}
  <div className="clinical-grid">
   <section className="panel case-list"><div className="section-title"><div><h2>Cases</h2><p>Held-out example studies and your uploads</p></div></div>
    <label className="upload-button clinical-upload"><Upload size={15}/> Upload MRI (DICOM ZIP / NIfTI)<input type="file" accept=".zip,.nii,.gz" disabled={busy} onChange={e=>{const f=e.target.files?.[0];if(f)upload(f);e.target.value=''}}/></label>
    {cases.map(c=><button key={c.id} className={'case-item'+(c.id===selected?' active':'')} onClick={()=>setSelected(c.id)}>
     <strong>{c.original_name||'Study '+short(c.id)}</strong><small>{c.split==='test'?'Example test study':'Uploaded'} · {c.series.length} sequence{c.series.length===1?'':'s'}</small>
     <span className={'status'+(c.latest_job?.status==='completed'?' success':'')}>{c.latest_job?c.latest_job.status:'Not analysed'}</span></button>)}
    {model?.available&&<div className="model-card"><h3>Deployed model</h3><p>Noise-aware pooled model · {model.seeds?.length}-seed ensemble · trained on {model.training_studies?.toLocaleString()} studies ({model.expert_target_studies} expert-labelled)</p>
     {model.reference_performance&&<p>Cross-validated macro AUC {model.reference_performance.macro_auc.toFixed(2)} on 58 expert studies</p>}<small>{model.id}</small></div>}
   </section>
   <div>{!current?<section className="panel"><div className="empty"><Layers size={30}/><p>Select or upload a case.</p></div></section>:<>
    <div className="study-heading"><h2>{current.original_name||'Study '+short(current.id)}</h2><span className="status">{current.split==='test'?'Example test · never used in training':'External upload'}</span></div>
    <Viewer key={current.id} series={current.series}/>
    {needsProtocol&&<section className="panel"><div className="section-title"><div><h2>Confirm sequence details</h2><p>Required before analysis: plane, fluid sensitivity and knee side for every sequence.</p></div></div>{current.series.map(s=><Protocol key={s.id} series={s} onSaved={load}/>)}</section>}
    <section className="panel analyze-bar"><div><h2>AI analysis</h2><p>{running?`${job?.message||'Queued'} · ${Math.round((job?.progress||0)*100)}%`:job?.status==='failed'?`Failed: ${job.error}`:result?`Analysed ${new Date(job!.created_at).toLocaleString()}`:'Encodes each sequence and applies the deployed model (about 10–30 seconds).'}</p>
     {running&&<div className="progress"><span style={{width:(job?.progress||0)*100+'%'}}/></div>}</div>
     <div className="analyze-actions">{result&&<button className="button secondary" onClick={()=>window.print()}><Printer size={15}/> Print report</button>}
      <button className="button" disabled={busy||running||needsProtocol||!model?.available} onClick={analyze}><Play size={15}/>{result?'Re-analyse':'Analyse study'}</button></div></section>
    {result&&<div className="print-area">
     <div className="print-only print-header"><h1>KneeWS knee MRI decision-support report</h1><p>Study {current.id} · generated {new Date(job!.created_at).toLocaleString()} · model {result.final_model_id}</p>
      <p><strong>Research software, not clinically validated. Findings must be confirmed by a qualified radiologist.</strong></p></div>
     <Findings result={result}/>
     <section className="panel sign-off"><label>Reviewing clinician (printed on report)<input value={clinician} onChange={e=>setClinician(e.target.value)} placeholder="Name"/></label><p className="print-only">Reviewed by: {clinician||'________________'} · Signature: ________________</p></section>
    </div>}
    {!result&&!running&&<section className="panel"><div className="empty"><FileText size={28}/><p>Run the analysis to see the 12-finding assessment.</p></div></section>}
   </>}</div>
  </div>
 </div>;
}
