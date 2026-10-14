import {chromium} from '../web/node_modules/playwright/index.mjs';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const origin=process.env.KNEEWS_UI_URL||'http://127.0.0.1:3200';
const fixture=process.env.KNEEWS_UI_FIXTURE||'upload-55b45b3b-532c-44e2-a968-288e00f64bad';
const output=new URL('../artifacts/ui-qa/',import.meta.url);
await mkdir(output,{recursive:true});
const browser=await chromium.launch({channel:'chrome',headless:true,timeout:20000});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
page.setDefaultTimeout(15000);
const errors=[];page.on('pageerror',error=>errors.push(error.message));
const checks=[];
async function check(name,run){await run();checks.push(name);console.log(JSON.stringify({check:name,status:'passed'}))}
try{
 const response=await page.request.get(`${origin}/api/studies/${fixture}`);
 assert.equal(response.status(),200);
 const study=await response.json();
 assert.equal(study.metadata.original_name,'synthetic-ui-fixture.nii.gz');
 assert.equal(study.split,'external');
 await page.goto(origin,{waitUntil:'domcontentloaded'});
 await check('Clinical workspace is the landing page with disclaimer and deployed model',async()=>{
  await page.getByRole('heading',{name:'Knee MRI assessment',level:1}).waitFor();
  await page.getByText(/not a medical device/).waitFor();
  await page.getByText('Deployed model',{exact:true}).waitFor();
  await page.locator('.case-item').first().waitFor();
 });
 await page.getByRole('button',{name:'Overview',exact:true}).click();
 await check('Overview loads real aggregate counts',async()=>{
  await page.getByText('Studies imported',{exact:true}).waitFor();
  await page.getByText('100.0%',{exact:true}).waitFor();
  await page.locator('table tbody tr').first().waitFor();
  await page.waitForFunction(()=>{const bar=document.querySelector('.dataset-panel .progress');return bar&&Math.abs(bar.getBoundingClientRect().width-bar.firstElementChild.getBoundingClientRect().width)<1});
 });
 await page.screenshot({path:new URL('overview-desktop.png',output).pathname,fullPage:true});
 for(const name of ['Pipeline','Experiments','Reliability','Label review']){
  await check(`${name} navigation`,async()=>{
   await page.getByRole('button',{name,exact:true}).click();
   await page.getByRole('heading',{name,exact:true,level:1}).waitFor();
  });
 }
 await page.getByRole('button',{name:'Study explorer',exact:true}).click();
 await page.getByRole('textbox',{name:'Search study ID'}).fill(fixture);
 await page.getByRole('button',{name:fixture.slice(-14),exact:true}).click();
 await check('Only synthetic fixture opened',async()=>{
  await page.getByText(/Synthetic display fixture; excluded from research/).waitFor();
  await page.getByRole('heading',{name:'Uploaded-study predictions'}).waitFor();
  await page.getByRole('heading',{name:'Review history',exact:true}).waitFor();
  await page.getByText('No saved reviews for this selection.',{exact:true}).waitFor();
 });
 await check('MRI loads and slice navigation works',async()=>{
  await page.waitForFunction(()=>{const image=document.querySelector('.image-area img');return image?.complete&&image.naturalWidth>0});
  const slider=page.getByRole('slider',{name:'Slice index'});
  await slider.fill('0');
  await page.getByRole('img',{name:'MRI slice 1',exact:true}).waitFor();
  await page.getByRole('combobox',{name:'Viewing plane'}).selectOption('0');
  await page.waitForFunction(()=>{const image=document.querySelector('.image-area img');return image?.complete&&image.naturalWidth>0});
 });
 await check('Synthetic protocol can be saved through the UI',async()=>{
  await page.getByLabel('Acquisition plane',{exact:true}).selectOption('Sagittal');
  await page.getByLabel('Fluid-sensitive sequence',{exact:true}).selectOption('1');
  await page.getByLabel('Knee side',{exact:true}).selectOption('L');
  await page.getByLabel('Protocol reviewer',{exact:true}).fill('Synthetic QA');
  await page.getByRole('button',{name:'Save sequence details'}).click();
  await page.getByText('Protocol saved.',{exact:true}).waitFor();
  const updated=await (await page.request.get(`${origin}/api/studies/${fixture}`)).json();
  assert.equal(updated.series[0].metadata.reviewer,'Synthetic QA');
  assert.equal(updated.series[0].metadata.protocol_reviewed,true);
 });
 await check('Validation errors persist across automatic refresh',async()=>{
  await page.getByRole('textbox',{name:'Reviewer',exact:true}).fill('Synthetic QA');
  await page.getByLabel('Reviewed state',{exact:true}).selectOption('present');
  await page.getByRole('button',{name:'Save review',exact:true}).click();
  const error=page.getByRole('alert').filter({hasText:'A mentioned state requires report evidence'});
  await error.waitFor();
  await page.waitForTimeout(6000);
  assert.equal(await error.isVisible(),true);
  await error.getByRole('button',{name:'Dismiss'}).click();
 });
 await page.screenshot({path:new URL('synthetic-study-desktop.png',output).pathname,fullPage:true});
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:new URL('synthetic-study-mobile.png',output).pathname,fullPage:true});
 await check('Mobile content fits viewport',async()=>{
  const dimensions=await page.evaluate(()=>({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));
  assert.ok(dimensions.scroll<=dimensions.width+1,JSON.stringify(dimensions));
 });
 assert.deepEqual(errors,[]);
 await check('Reliability stage follows completed pooled experiments',async()=>{
  // Browser-only response fixtures: never insert synthetic experiments into the database.
  let experiment={id:'browser-status-fixture',name:'Browser status fixture',status:'running',arms:['pooled']};
  await page.route('**/api/experiments',route=>route.fulfill({json:[experiment]}));
  for(const [status,arms,expected] of [
   ['running',['pooled'],'Pending'],
   ['completed',['binary'],'Pending'],
   ['completed',['binary','pooled'],'Available']
  ]){
   experiment={...experiment,status,arms};
   await page.reload({waitUntil:'networkidle'});
   await page.getByRole('heading',{name:'Knee MRI assessment',level:1}).waitFor();
   await page.getByRole('button',{name:'Overview',exact:true}).click();
   await page.getByRole('heading',{name:'From reports to reliable evidence.',level:1}).waitFor();
   const stage=page.locator('.stage').filter({has:page.getByRole('heading',{name:'Reliability estimation',exact:true})});
   await stage.locator('.status').filter({hasText:expected}).waitFor();
  }
  await page.unroute('**/api/experiments');
 });
 assert.deepEqual(errors,[]);
 await writeFile(new URL('result.json',output),JSON.stringify({status:'passed',fixture,checks,browser:await browser.version(),pageErrors:errors},null,2));
 console.log(JSON.stringify({status:'passed',checks,output:output.pathname}));
}catch(error){
 await writeFile(new URL('result.json',output),JSON.stringify({status:'failed',checks,error:String(error),pageErrors:errors},null,2));
 console.error(String(error));process.exitCode=1;
}finally{await browser.close()}
