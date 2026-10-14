export async function getJSON<T>(path:string,options?:RequestInit):Promise<T>{
  const response=await fetch('/api'+path,options);
  if(!response.ok){const value=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail));}
  return response.json();
}
