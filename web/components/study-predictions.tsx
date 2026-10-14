'use client';
import {useEffect,useState} from 'react';
import {getJSON} from '../lib/api';

type Experiment={id:string;name:string;status:string;arms:string[]};
type Prediction={study_id:string;seeds:number;findings:{finding:string;probability:number;seed_min:number;seed_max:number;review_required:boolean;reasons:string[]}[]};
const percent=(value:number)=>(value*100).toFixed(1)+'%';

export function StudyPredictions({studyId,isGold}:{studyId:string;isGold:boolean}){
 const [experiments,setExperiments]=useState<Experiment[]>([]),[selected,setSelected]=useState(''),[arm,setArm]=useState('pooled'),[result,setResult]=useState<Prediction|null>(null),[error,setError]=useState(''),[loading,setLoading]=useState(false);
 useEffect(()=>{let active=true;async function load(){try{const data=await getJSON<Experiment[]>('/experiments');if(!active)return;const completed=data.filter(x=>x.status==='completed');setExperiments(completed);setSelected(current=>completed.some(x=>x.id===current)?current:completed[0]?.id||'')}catch(e){if(active)setError((e as Error).message)}}load();const timer=setInterval(load,15000);return()=>{active=false;clearInterval(timer)}},[]);
 const experiment=experiments.find(x=>x.id===selected);
 const selectedArm=experiment?.arms.includes(arm)?arm:experiment?.arms[0]||'';
 useEffect(()=>{let active=true;setResult(null);setError('');if(!selected||!selectedArm||!isGold){setLoading(false);return;}setLoading(true);getJSON<Prediction>(`/experiments/${encodeURIComponent(selected)}/studies/${encodeURIComponent(studyId)}?arm=${encodeURIComponent(selectedArm)}`).then(data=>{if(active)setResult(data)}).catch(e=>{if(active)setError((e as Error).message)}).finally(()=>{if(active)setLoading(false)});return()=>{active=false}},[studyId,isGold,selected,selectedArm]);
 return <section className="panel"><div className="section-title"><div><h2>Image model findings</h2><p>Held-out predictions from completed research experiments.</p></div></div>
 {error&&<div role="alert" className="alert error">{error}</div>}
 {!isGold?<div className="empty">This study has no expert-cohort held-out prediction. Predictions for new uploads are not available yet.</div>:!experiments.length?<div className="empty">The image experiments have not completed. Findings will appear here after evaluation.</div>:<>
 <div className="filters"><label>Experiment<select value={selected} onChange={e=>setSelected(e.target.value)}>{experiments.map(x=><option key={x.id} value={x.id}>{x.name}</option>)}</select></label><label>Training arm<select value={selectedArm} onChange={e=>setArm(e.target.value)}>{experiment?.arms.map(x=><option key={x} value={x}>{x}</option>)}</select></label></div>
 {loading&&<p role="status">Loading held-out predictions…</p>}
 {result&&<><p className="panel-note">Mean probability across {result.seeds} seeds. The seed range shows training variability and is not a confidence interval. These review thresholds have not been clinically validated.</p><div className="table-wrap"><table><thead><tr><th>FINDING</th><th>PROBABILITY</th><th>SEED RANGE</th><th>REVIEW DECISION</th></tr></thead><tbody>{result.findings.map(x=><tr key={x.finding}><td>{x.finding}</td><td>{percent(x.probability)}</td><td>{percent(x.seed_min)} – {percent(x.seed_max)}</td><td><span className="status">{x.review_required?'Abstain · review required':'Outside review band'}</span>{x.reasons.length>0&&<p>{x.reasons.join('; ')}</p>}</td></tr>)}</tbody></table></div></>}
 </>}
 </section>
}
