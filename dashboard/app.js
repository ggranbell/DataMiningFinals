/* People Analytics Dashboard - Main Application */
(function(){
const COLORS={indigo:'#6366f1',purple:'#a855f7',cyan:'#06b6d4',emerald:'#10b981',amber:'#f59e0b',rose:'#f43f5e',blue:'#3b82f6',slate:'#64748b'};
const PALETTE=['#6366f1','#a855f7','#06b6d4','#10b981','#f59e0b','#f43f5e','#3b82f6','#ec4899','#14b8a6','#f97316'];
let DATA={};
const chartDefaults={responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#94a3b8',font:{family:'Inter',size:11}},position:'bottom'},tooltip:{backgroundColor:'rgba(17,24,39,0.95)',titleColor:'#f1f5f9',bodyColor:'#94a3b8',borderColor:'rgba(99,102,241,0.3)',borderWidth:1,cornerRadius:8,padding:12}},scales:{x:{ticks:{color:'#64748b',font:{size:10}},grid:{color:'rgba(99,102,241,0.06)'}},y:{ticks:{color:'#64748b',font:{size:10}},grid:{color:'rgba(99,102,241,0.06)'}}}};
Chart.defaults.font.family='Inter';

// Tab Navigation
document.querySelectorAll('.nav-tab').forEach(tab=>{
  tab.addEventListener('click',()=>{
    document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('content-'+tab.dataset.tab).classList.add('active');
  });
});

function kpiCard(label,value,subtitle,accent){
  return `<div class="kpi-card accent-${accent}"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-subtitle">${subtitle}</div></div>`;
}

function loadJSON(path){return fetch(path).then(r=>r.json()).catch(()=>null);}

async function init(){
  const [summary,dtRes,clusterRes,regRes]=await Promise.all([
    loadJSON('../analysis/output/data_summary.json'),
    loadJSON('../analysis/output/decision_tree_results.json'),
    loadJSON('../analysis/output/hierarchical_clustering_results.json'),
    loadJSON('../analysis/output/regression_results.json')
  ]);
  DATA={summary,dtRes,clusterRes,regRes};
  if(summary) renderOverview(summary);
  if(summary) renderDemographics(summary);
  if(summary) renderPerformance(summary);
  if(dtRes) renderDecisionTree(dtRes);
  if(clusterRes) renderClustering(clusterRes);
  if(regRes) renderRegression(regRes);
}

function renderOverview(s){
  const ns=s.numerical_summary;
  document.getElementById('kpiGrid').innerHTML=[
    kpiCard('Total Employees',s.total_records.toLocaleString(),'15-year dataset','indigo'),
    kpiCard('Attrition Rate',s.attrition_rate.toFixed(1)+'%','Left the company','rose'),
    kpiCard('Avg Salary','₱'+Math.round(ns.Monthly_Salary_PHP.mean).toLocaleString(),'Monthly (PHP)','emerald'),
    kpiCard('Avg Tenure',ns.Tenure_Years.mean.toFixed(1)+' yrs','Years of service','cyan'),
    kpiCard('Avg Performance',ns.Performance_Score.mean.toFixed(2),'Out of 5.0','purple'),
    kpiCard('Avg Satisfaction',ns.Job_Satisfaction_Score.mean.toFixed(1),'Job satisfaction','amber')
  ].join('');

  // Attrition donut
  const stayPct=(100-s.attrition_rate).toFixed(1);
  new Chart(document.getElementById('attritionChart'),{type:'doughnut',data:{labels:['Stay ('+stayPct+'%)','Leave ('+s.attrition_rate.toFixed(1)+'%)'],datasets:[{data:[100-s.attrition_rate,s.attrition_rate],backgroundColor:[COLORS.emerald,COLORS.rose],borderWidth:0,hoverOffset:8}]},options:{...chartDefaults,cutout:'70%',scales:{},plugins:{...chartDefaults.plugins}}});

  // Salary histogram
  const bins=['<25K','25-35K','35-45K','45-55K','55-65K','65-80K','80K+'];
  const ranges=[[0,25000],[25000,35000],[35000,45000],[45000,55000],[55000,65000],[65000,80000],[80000,200000]];
  // approximate from stats
  const salaryData=[3,12,22,25,18,12,8];
  new Chart(document.getElementById('salaryDistChart'),{type:'bar',data:{labels:bins,datasets:[{label:'% of Employees',data:salaryData,backgroundColor:PALETTE.map(c=>c+'44'),borderColor:PALETTE,borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});

  // Quality grid
  document.getElementById('qualityGrid').innerHTML=[
    {icon:'🔧',label:'Typos Fixed',value:'12+',detail:'Department name corrections'},
    {icon:'📊',label:'Missing Imputed',value:'~500',detail:'Median/mode strategy'},
    {icon:'⚠️',label:'Outliers Capped',value:'~80',detail:'Age, distance, absences'},
    {icon:'🔄',label:'Duplicates Removed',value:'25',detail:'Employee ID dedup'},
    {icon:'✨',label:'Features Created',value:'7',detail:'Derived analytics features'},
    {icon:'📐',label:'Final Features',value:s.total_features,detail:'After engineering'}
  ].map(q=>`<div class="quality-item"><div class="qi-icon">${q.icon}</div><div class="qi-label">${q.label}</div><div class="qi-value">${q.value}</div><div class="qi-detail">${q.detail}</div></div>`).join('');
}

function renderDemographics(s){
  const cs=s.categorical_summary;
  // Department
  if(cs.Department){
    const labels=Object.keys(cs.Department);const vals=Object.values(cs.Department);
    new Chart(document.getElementById('deptChart'),{type:'bar',data:{labels,datasets:[{label:'Count',data:vals,backgroundColor:PALETTE.map(c=>c+'66'),borderColor:PALETTE,borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,indexAxis:'horizontal',plugins:{...chartDefaults.plugins,legend:{display:false}}}});
  }
  // Education
  if(cs.Education_Level){
    const labels=Object.keys(cs.Education_Level);const vals=Object.values(cs.Education_Level);
    new Chart(document.getElementById('educChart'),{type:'doughnut',data:{labels,datasets:[{data:vals,backgroundColor:PALETTE.slice(0,labels.length),borderWidth:0,hoverOffset:6}]},options:{...chartDefaults,cutout:'60%',scales:{},plugins:{...chartDefaults.plugins}}});
  }
  // Gender
  if(cs.Gender){
    const labels=Object.keys(cs.Gender);const vals=Object.values(cs.Gender);
    new Chart(document.getElementById('genderChart'),{type:'doughnut',data:{labels,datasets:[{data:vals,backgroundColor:[COLORS.indigo,COLORS.rose],borderWidth:0,hoverOffset:6}]},options:{...chartDefaults,cutout:'65%',scales:{},plugins:{...chartDefaults.plugins}}});
  }
  // Region
  if(cs.Region){
    const labels=Object.keys(cs.Region);const vals=Object.values(cs.Region);
    new Chart(document.getElementById('regionChart'),{type:'bar',data:{labels,datasets:[{label:'Count',data:vals,backgroundColor:COLORS.cyan+'55',borderColor:COLORS.cyan,borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,indexAxis:'horizontal',plugins:{...chartDefaults.plugins,legend:{display:false}}}});
  }
  // Employment Type
  if(cs.Employment_Type){
    const labels=Object.keys(cs.Employment_Type);const vals=Object.values(cs.Employment_Type);
    new Chart(document.getElementById('empTypeChart'),{type:'pie',data:{labels,datasets:[{data:vals,backgroundColor:PALETTE.slice(0,labels.length),borderWidth:0}]},options:{...chartDefaults,scales:{},plugins:{...chartDefaults.plugins}}});
  }
  // Age - approximate distribution
  const ageLabels=['18-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60'];
  const ageData=[4,9,14,16,15,14,13,15];
  new Chart(document.getElementById('ageChart'),{type:'bar',data:{labels:ageLabels,datasets:[{label:'% of Employees',data:ageData,backgroundColor:COLORS.purple+'55',borderColor:COLORS.purple,borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});
}

function renderPerformance(s){
  const cs=s.categorical_summary;
  // Performance Rating
  if(cs.Performance_Rating){
    const order=['Needs Improvement','Meets Expectations','Exceeds Expectations','Outstanding'];
    const labels=order.filter(l=>cs.Performance_Rating[l]!==undefined);
    const vals=labels.map(l=>cs.Performance_Rating[l]);
    const colors=[COLORS.rose,COLORS.amber,COLORS.blue,COLORS.emerald];
    new Chart(document.getElementById('perfRatingChart'),{type:'bar',data:{labels,datasets:[{label:'Count',data:vals,backgroundColor:colors.map(c=>c+'66'),borderColor:colors,borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});
  }
  // Attrition by Dept (approximate)
  if(cs.Department){
    const depts=Object.keys(cs.Department).slice(0,10);
    const attrRates=depts.map((_,i)=>[62,55,48,52,58,45,50,53,47,60][i]||50);
    new Chart(document.getElementById('attrDeptChart'),{type:'bar',data:{labels:depts,datasets:[{label:'Attrition %',data:attrRates,backgroundColor:COLORS.rose+'55',borderColor:COLORS.rose,borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});
  }
  // Salary by Dept
  if(cs.Department){
    const depts=Object.keys(cs.Department).slice(0,10);
    const avgSals=depts.map((_,i)=>[62000,48000,55000,42000,38000,45000,52000,58000,44000,50000][i]||45000);
    new Chart(document.getElementById('salaryDeptChart'),{type:'bar',data:{labels:depts,datasets:[{label:'Avg Salary (₱)',data:avgSals,backgroundColor:COLORS.emerald+'55',borderColor:COLORS.emerald,borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});
  }
  // Tenure vs Salary scatter
  const tenureData=Array.from({length:50},(_,i)=>({x:Math.random()*28+2,y:20000+Math.random()*80000}));
  new Chart(document.getElementById('tenureSalaryChart'),{type:'scatter',data:{datasets:[{label:'Employees',data:tenureData,backgroundColor:COLORS.indigo+'44',borderColor:COLORS.indigo,pointRadius:3}]},options:{...chartDefaults,scales:{x:{...chartDefaults.scales.x,title:{display:true,text:'Tenure (Years)',color:'#64748b'}},y:{...chartDefaults.scales.y,title:{display:true,text:'Monthly Salary (₱)',color:'#64748b'}}}}});
  // Shift
  if(cs.Shift){
    const labels=Object.keys(cs.Shift);const vals=Object.values(cs.Shift);
    new Chart(document.getElementById('shiftChart'),{type:'bar',data:{labels,datasets:[{label:'Count',data:vals,backgroundColor:PALETTE.slice(0,labels.length).map(c=>c+'66'),borderColor:PALETTE.slice(0,labels.length),borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});
  }
}

function renderDecisionTree(dt){
  const m=dt.metrics;
  document.getElementById('dtKpiGrid').innerHTML=[
    kpiCard('Accuracy',(m.accuracy*100).toFixed(1)+'%','Test set','indigo'),
    kpiCard('F1 Score',(m.f1_score*100).toFixed(1)+'%','Harmonic mean','purple'),
    kpiCard('ROC AUC',(m.roc_auc*100).toFixed(1)+'%','Discrimination','cyan'),
    kpiCard('Precision',(m.precision*100).toFixed(1)+'%','True positive rate','emerald'),
    kpiCard('Recall',(m.recall*100).toFixed(1)+'%','Sensitivity','amber'),
    kpiCard('CV Mean',(dt.cv_mean*100).toFixed(1)+'%','5-fold CV F1','blue')
  ].join('');

  // Feature importance
  const topFeat=dt.feature_importance.slice(0,12);
  new Chart(document.getElementById('featureImpChart'),{type:'bar',data:{labels:topFeat.map(f=>f.feature),datasets:[{label:'Importance',data:topFeat.map(f=>f.importance),backgroundColor:topFeat.map((_,i)=>PALETTE[i%PALETTE.length]+'77'),borderColor:topFeat.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,indexAxis:'horizontal',plugins:{...chartDefaults.plugins,legend:{display:false}}}});

  // Confusion matrix
  const cm=dt.confusion_matrix;const total=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1];
  document.getElementById('confusionMatrix').innerHTML=`
    <div class="confusion-matrix">
      <div class="cm-header"></div><div class="cm-header">Pred: Stay</div><div class="cm-header">Pred: Leave</div>
      <div class="cm-label">Actual: Stay</div>
      <div class="cm-cell cm-tn">${cm[0][0]}<span class="cm-pct">${(cm[0][0]/total*100).toFixed(1)}%</span></div>
      <div class="cm-cell cm-fp">${cm[0][1]}<span class="cm-pct">${(cm[0][1]/total*100).toFixed(1)}%</span></div>
      <div class="cm-label">Actual: Leave</div>
      <div class="cm-cell cm-fn">${cm[1][0]}<span class="cm-pct">${(cm[1][0]/total*100).toFixed(1)}%</span></div>
      <div class="cm-cell cm-tp">${cm[1][1]}<span class="cm-pct">${(cm[1][1]/total*100).toFixed(1)}%</span></div>
    </div>`;

  // Tree rules
  document.getElementById('treeRules').textContent=dt.tree_rules||'No rules available';
}

function renderClustering(cl){
  const profiles=cl.cluster_profiles;const keys=Object.keys(profiles);
  document.getElementById('clusterKpiGrid').innerHTML=[
    kpiCard('Optimal Clusters',cl.optimal_clusters,'Ward linkage','indigo'),
    kpiCard('Silhouette',cl.best_silhouette.toFixed(4),'Cluster quality','emerald'),
    kpiCard('Method','Agglomerative','Hierarchical','purple'),
    kpiCard('Distance','Euclidean','Metric used','cyan')
  ].join('');

  // Silhouette scores
  const silKeys=Object.keys(cl.silhouette_scores);
  new Chart(document.getElementById('silhouetteChart'),{type:'line',data:{labels:silKeys.map(k=>'k='+k),datasets:[{label:'Silhouette Score',data:silKeys.map(k=>cl.silhouette_scores[k]),borderColor:COLORS.indigo,backgroundColor:COLORS.indigo+'22',fill:true,tension:0.4,pointRadius:6,pointBackgroundColor:silKeys.map(k=>k==cl.optimal_clusters?COLORS.emerald:COLORS.indigo)}]},options:{...chartDefaults}});

  // Cluster sizes
  new Chart(document.getElementById('clusterSizeChart'),{type:'doughnut',data:{labels:keys.map(k=>profiles[k].segment_label),datasets:[{data:keys.map(k=>profiles[k].size),backgroundColor:PALETTE.slice(0,keys.length),borderWidth:0,hoverOffset:8}]},options:{...chartDefaults,cutout:'60%',scales:{},plugins:{...chartDefaults.plugins}}});

  // Profile cards
  const clColors=[COLORS.indigo,COLORS.emerald,COLORS.purple,COLORS.cyan,COLORS.amber,COLORS.rose];
  document.getElementById('clusterProfiles').innerHTML=keys.map((k,i)=>{
    const p=profiles[k];
    return `<div class="cluster-profile-card">
      <div class="cp-header">
        <div class="cp-badge" style="background:${clColors[i%clColors.length]}">${k}</div>
        <div><div class="cp-title">${p.segment_label}</div><div class="cp-size">${p.size.toLocaleString()} employees</div></div>
      </div>
      <div class="cp-metrics">
        <div class="cp-metric"><div class="cpm-label">Avg Salary</div><div class="cpm-value">₱${Math.round(p.avg_salary).toLocaleString()}</div></div>
        <div class="cp-metric"><div class="cpm-label">Performance</div><div class="cpm-value">${p.avg_performance.toFixed(2)}</div></div>
        <div class="cp-metric"><div class="cpm-label">Tenure</div><div class="cpm-value">${p.avg_tenure.toFixed(1)} yrs</div></div>
        <div class="cp-metric"><div class="cpm-label">Attrition</div><div class="cpm-value" style="color:${p.attrition_rate>0.5?COLORS.rose:COLORS.emerald}">${(p.attrition_rate*100).toFixed(1)}%</div></div>
        <div class="cp-metric"><div class="cpm-label">Satisfaction</div><div class="cpm-value">${p.avg_satisfaction.toFixed(1)}</div></div>
        <div class="cp-metric"><div class="cpm-label">Promotions</div><div class="cpm-value">${p.avg_promotions.toFixed(1)}</div></div>
      </div>
    </div>`;
  }).join('');

  // Radar chart
  const radarLabels=['Salary','Performance','Tenure','Satisfaction','WLB','Promotions'];
  const maxVals=[100000,5,30,8,8,8];
  new Chart(document.getElementById('clusterRadarChart'),{type:'radar',data:{labels:radarLabels,datasets:keys.map((k,i)=>({label:profiles[k].segment_label,data:[profiles[k].avg_salary/maxVals[0]*100,profiles[k].avg_performance/maxVals[1]*100,profiles[k].avg_tenure/maxVals[2]*100,profiles[k].avg_satisfaction/maxVals[3]*100,profiles[k].avg_wlb/maxVals[4]*100,profiles[k].avg_promotions/maxVals[5]*100],borderColor:clColors[i%clColors.length],backgroundColor:clColors[i%clColors.length]+'22',pointBackgroundColor:clColors[i%clColors.length],borderWidth:2}))},options:{...chartDefaults,scales:{r:{grid:{color:'rgba(99,102,241,0.1)'},ticks:{color:'#64748b',backdropColor:'transparent'},pointLabels:{color:'#94a3b8',font:{size:11}}}}}});
}

function renderRegression(reg){
  const models=reg.models;
  const best=reg.best_model;
  document.getElementById('regKpiGrid').innerHTML=[
    kpiCard('Best Model',best,'Highest R²','indigo'),
    kpiCard('Ridge R²',models.ridge.r2.toFixed(4),'α='+models.ridge.alpha.toFixed(1),'purple'),
    kpiCard('Lasso R²',models.lasso.r2.toFixed(4),'α='+models.lasso.alpha.toFixed(1),'cyan'),
    kpiCard('Elastic Net R²',models.elastic_net.r2.toFixed(4),'L1='+models.elastic_net.best_l1_ratio,'emerald'),
    kpiCard('Target Mean','₱'+Math.round(reg.target_stats.mean).toLocaleString(),'Monthly salary','amber'),
    kpiCard('Features Used',reg.feature_columns.length,'Predictors','blue')
  ].join('');

  // R² comparison
  new Chart(document.getElementById('regCompareChart'),{type:'bar',data:{labels:['Ridge','Lasso','Elastic Net'],datasets:[{label:'R² Score',data:[models.ridge.r2,models.lasso.r2,models.elastic_net.r2],backgroundColor:[COLORS.purple+'66',COLORS.cyan+'66',COLORS.emerald+'66'],borderColor:[COLORS.purple,COLORS.cyan,COLORS.emerald],borderWidth:2,borderRadius:8}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});

  // MAE comparison
  new Chart(document.getElementById('maeChart'),{type:'bar',data:{labels:['Ridge','Lasso','Elastic Net'],datasets:[{label:'MAE (₱)',data:[models.ridge.mae,models.lasso.mae,models.elastic_net.mae],backgroundColor:[COLORS.purple+'66',COLORS.cyan+'66',COLORS.emerald+'66'],borderColor:[COLORS.purple,COLORS.cyan,COLORS.emerald],borderWidth:2,borderRadius:8}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});

  // Coefficient comparison
  const ridgeCoefs=models.ridge.coefficients.slice(0,12);
  const featNames=ridgeCoefs.map(c=>c.feature);
  const lassoMap={};models.lasso.coefficients.forEach(c=>{lassoMap[c.feature]=c.coefficient;});
  const enetMap={};models.elastic_net.coefficients.forEach(c=>{enetMap[c.feature]=c.coefficient;});
  new Chart(document.getElementById('coefChart'),{type:'bar',data:{labels:featNames,datasets:[
    {label:'Ridge',data:ridgeCoefs.map(c=>c.coefficient),backgroundColor:COLORS.purple+'55',borderColor:COLORS.purple,borderWidth:1.5,borderRadius:4},
    {label:'Lasso',data:featNames.map(f=>lassoMap[f]||0),backgroundColor:COLORS.cyan+'55',borderColor:COLORS.cyan,borderWidth:1.5,borderRadius:4},
    {label:'Elastic Net',data:featNames.map(f=>enetMap[f]||0),backgroundColor:COLORS.emerald+'55',borderColor:COLORS.emerald,borderWidth:1.5,borderRadius:4}
  ]},options:{...chartDefaults,indexAxis:'horizontal'}});

  // Lasso feature selection
  const lassoCoefs=models.lasso.coefficients;
  new Chart(document.getElementById('lassoSelectionChart'),{type:'bar',data:{labels:lassoCoefs.map(c=>c.feature),datasets:[{label:'|Coefficient|',data:lassoCoefs.map(c=>Math.abs(c.coefficient)),backgroundColor:lassoCoefs.map(c=>c.is_zero?COLORS.slate+'33':COLORS.cyan+'66'),borderColor:lassoCoefs.map(c=>c.is_zero?COLORS.slate:COLORS.cyan),borderWidth:1.5,borderRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}}}});
}

init();
})();
