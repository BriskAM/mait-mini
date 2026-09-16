async function loadDashboard(){
  const [edaRes, resultRes] = await Promise.all([fetch('data/eda_summary.json'), fetch('data/results.json')]);
  const eda = await edaRes.json(); const results = await resultRes.json();
  const fmt = n => Number(n).toLocaleString();
  document.querySelector('#metrics').innerHTML = [
    ['51,839','total images'],[fmt(eda.train_images),'training images'],[fmt(eda.test_images),'official test images'],[String(eda.classes),'sign classes']
  ].map(([v,l])=>`<div class="metric"><strong>${v}</strong><span>${l}</span></div>`).join('');
  document.querySelector('#model-cards').innerHTML = (results.models || []).map(m => `<article class="model-card"><h3>${m.name.replaceAll('_',' ')}</h3><div class="score-row"><span>Accuracy</span><strong>${(m.accuracy*100).toFixed(2)}%</strong></div><div class="score-row"><span>Macro F1</span><strong>${(m.macro_f1*100).toFixed(2)}%</strong></div></article>`).join('') || '<p>No completed model results yet.</p>';
}
const imageInput = document.querySelector('#image-input');
const preview = document.querySelector('#preview');
const predictionOutput = document.querySelector('#prediction-output');
const gradcam = document.querySelector('#gradcam');
const demoStatus = document.querySelector('#demo-status');
let selectedFile = null; let stream = null;
imageInput?.addEventListener('change', event => { selectedFile = event.target.files[0]; if (selectedFile) { preview.src = URL.createObjectURL(selectedFile); preview.hidden = false; demoStatus.textContent = 'Image ready. Click Predict image.'; } });
async function predict(file){ const form = new FormData(); form.append('image', file); demoStatus.textContent = 'Running model and Grad-CAM...'; const response = await fetch('/predict',{method:'POST',body:form}); const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Prediction failed'); predictionOutput.innerHTML = `<strong>Class ${result.class_id} · ${(result.confidence*100).toFixed(2)}%</strong><span>Top predictions: ${result.top_k.map(x=>`Class ${x.class_id} ${(x.confidence*100).toFixed(1)}%`).join(' · ')}</span>`; gradcam.src = result.gradcam_url + '?t=' + Date.now(); gradcam.hidden = false; demoStatus.textContent = `Explanation generated from layer ${result.cam_layer}.`; }
document.querySelector('#predict-button')?.addEventListener('click', async () => { const video = document.querySelector('#camera'); if (!video.hidden && video.videoWidth) { const canvas = document.querySelector('#capture'); canvas.width = video.videoWidth; canvas.height = video.videoHeight; canvas.getContext('2d').drawImage(video, 0, 0); canvas.toBlob(blob => predict(new File([blob], 'camera.png', {type:'image/png'})).catch(error => { demoStatus.textContent = error.message; }),'image/png'); return; } if (!selectedFile) { demoStatus.textContent = 'Choose an image first.'; return; } try { await predict(selectedFile); } catch (error) { demoStatus.textContent = error.message; } });
document.querySelector('#camera-button')?.addEventListener('click', async () => { try { stream = await navigator.mediaDevices.getUserMedia({video:true}); const video = document.querySelector('#camera'); video.srcObject = stream; video.hidden = false; document.querySelector('#camera-button').hidden = true; document.querySelector('#stop-camera').hidden = false; demoStatus.textContent = 'Camera active. Click Predict image to classify the current frame.'; } catch (error) { demoStatus.textContent = 'Camera access unavailable: ' + error.message; } });
document.querySelector('#stop-camera')?.addEventListener('click', () => { stream?.getTracks().forEach(track => track.stop()); document.querySelector('#camera').hidden = true; document.querySelector('#camera-button').hidden = false; document.querySelector('#stop-camera').hidden = true; });
loadDashboard().catch(error => { document.querySelector('#metrics').innerHTML = `<div class="metric"><strong>Dashboard data unavailable</strong><span>${error.message}</span></div>`; });
