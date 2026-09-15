async function loadDashboard(){
  const [edaRes, resultRes] = await Promise.all([fetch('data/eda_summary.json'), fetch('data/results.json')]);
  const eda = await edaRes.json(); const results = await resultRes.json();
  const fmt = n => Number(n).toLocaleString();
  document.querySelector('#metrics').innerHTML = [
    ['51,839','total images'],[fmt(eda.train_images),'training images'],[fmt(eda.test_images),'official test images'],[String(eda.classes),'sign classes']
  ].map(([v,l])=>`<div class="metric"><strong>${v}</strong><span>${l}</span></div>`).join('');
  document.querySelector('#model-cards').innerHTML = (results.models || []).map(m => `<article class="model-card"><h3>${m.name.replaceAll('_',' ')}</h3><div class="score-row"><span>Accuracy</span><strong>${(m.accuracy*100).toFixed(2)}%</strong></div><div class="score-row"><span>Macro F1</span><strong>${(m.macro_f1*100).toFixed(2)}%</strong></div></article>`).join('') || '<p>No completed model results yet.</p>';
}
loadDashboard().catch(error => { document.querySelector('#metrics').innerHTML = `<div class="metric"><strong>Dashboard data unavailable</strong><span>${error.message}</span></div>`; });
