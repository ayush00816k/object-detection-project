let selectedImage = null;
let chartInstance = null;
let liveInterval = null;
const { jsPDF } = window.jspdf;

document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('preview');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const useWebcamBtn = document.getElementById('useWebcamBtn');
  const webcam = document.getElementById('webcam');
  const captureBtn = document.getElementById('captureBtn');
  const liveToggleBtn = document.getElementById('liveToggleBtn');
  const videoOverlay = document.getElementById('videoOverlay');
  const uploadSection = document.getElementById('uploadSection');
  const results = document.getElementById('results');
  const resultImage = document.getElementById('resultImage');
  const detectedObject = document.getElementById('detectedObject');
  const detectedConfidence = document.getElementById('detectedConfidence');
  const modelScoreList = document.getElementById('modelScoreList');
  const bestModel = document.getElementById('bestModel');
  const bestModelStats = document.getElementById('bestModelStats');
  const analysisTime = document.getElementById('analysisTime');
  const modelsEvaluated = document.getElementById('modelsEvaluated');
  const summaryText = document.getElementById('summaryText');
  const insightsList = document.getElementById('insightsList');
  const testAnotherBtn = document.getElementById('testAnotherBtn');
  const downloadReportBtn = document.getElementById('downloadReportBtn');
  const uploadContainer = document.getElementById('uploadContainer');
  const orDivider = document.getElementById('orDivider');
  const webcamContainer = document.getElementById('webcamContainer');
  const classificationSummary = document.getElementById('classificationSummary');

  // Handle file upload
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        preview.src = event.target.result;
        preview.classList.remove('hidden');
        selectedImage = file;
        enableAnalyzeBtn();
        hideOtherOptions('upload');
      };
      reader.readAsDataURL(file);
    }
  });

  // Handle drag and drop
  const uploadBox = document.getElementById('uploadBox');
  if (uploadBox) {
    uploadBox.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadBox.classList.add('border-blue-500', 'bg-blue-100');
    });
    uploadBox.addEventListener('dragleave', () => {
      uploadBox.classList.remove('border-blue-500', 'bg-blue-100');
    });
    uploadBox.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadBox.classList.remove('border-blue-500', 'bg-blue-100');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
      }
    });
  }

  // Handle webcam
  useWebcamBtn.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      webcam.srcObject = stream;
      webcamContainer.classList.remove('hidden');
      hideOtherOptions('webcam');
      analyzeBtn.classList.add('hidden'); // Hide for webcam, use capture/live instead
    } catch (err) {
      alert('Error accessing webcam: ' + err.message);
    }
  });

  // Capture photo from webcam
  captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    canvas.getContext('2d').drawImage(webcam, 0, 0);
    preview.src = canvas.toDataURL('image/jpeg');
    preview.classList.remove('hidden');
    selectedImage = dataURLToFile(preview.src, 'captured.jpg');
    enableAnalyzeBtn();
  });

  // Live detection toggle
  liveToggleBtn.addEventListener('click', () => {
    if (liveInterval) {
      clearInterval(liveInterval);
      liveInterval = null;
      videoOverlay.classList.add('hidden');
      liveToggleBtn.textContent = 'Start Live Detection';
      liveToggleBtn.classList.remove('bg-red-500');
      liveToggleBtn.classList.add('bg-green-500');
    } else {
      liveInterval = setInterval(sendWebcamFrame, 500);
      liveToggleBtn.textContent = 'Stop Live Detection';
      liveToggleBtn.classList.remove('bg-green-500');
      liveToggleBtn.classList.add('bg-red-500');
    }
  });

  // Send webcam frame for live prediction
  async function sendWebcamFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    canvas.getContext('2d').drawImage(webcam, 0, 0);
    const frameData = canvas.toDataURL('image/jpeg');

    try {
      const response = await fetch('/detect-webcam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameData })
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Overlay live prediction
      videoOverlay.textContent = `${data.object} (${data.confidence}%) - Best: ${data.best_model}`;
      videoOverlay.classList.remove('hidden');
    } catch (err) {
      console.error('Live detection error:', err);
      videoOverlay.textContent = 'Error detecting';
      videoOverlay.classList.remove('hidden');
    }
  }

  // Analyze button
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('/detect', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Show results
      uploadSection.classList.add('hidden');
      results.classList.remove('hidden');

      // Populate results
      resultImage.src = preview.src;
      detectedObject.textContent = data.object;
      detectedConfidence.textContent = `${data.confidence}%`;
      analysisTime.textContent = `${data.time}s`;
      modelsEvaluated.textContent = data.evaluated;
      bestModel.textContent = data.best_model;
      bestModelStats.textContent = `Confidence: ${data.confidence}%`;

      // Model scores list (highlight best)
      modelScoreList.innerHTML = '';
      Object.entries(data.scores).forEach(([model, score]) => {
        const div = document.createElement('div');
        div.className = 'flex justify-between items-center p-2 rounded bg-gray-50';
        div.innerHTML = `<span class="font-medium ${model === data.best_model ? 'text-green-600' : ''}">${model}</span><span class="text-gray-600">${score}%</span>`;
        modelScoreList.appendChild(div);
      });

      // Chart
      if (chartInstance) chartInstance.destroy();
      chartInstance = new Chart(document.getElementById('chart'), {
        type: 'bar',
        data: {
          labels: Object.keys(data.scores),
          datasets: [{
            label: 'Confidence (%)',
            data: Object.values(data.scores),
            backgroundColor: Object.keys(data.scores).map(model => model === data.best_model ? 'rgba(34,197,94,0.7)' : 'rgba(59,130,246,0.7)'),
          }]
        },
        options: {
          scales: { y: { beginAtZero: true, max: 100 } },
          plugins: { legend: { display: false } }
        }
      });

      // Summary text
      classificationSummary.textContent = `The system evaluated the image using ${data.evaluated} models. The highest confidence prediction is "${data.object}" from ${data.best_model}.`;
      summaryText.textContent = `Overall, deep learning models performed well, with ${data.best_model} leading. Processing took ${data.time} seconds.`;

      // Insights (example, customize based on data)
      insightsList.innerHTML = '';
      const insights = [
        `Best model (${data.best_model}) achieved ${data.confidence}% confidence.`,
        'YOLO is great for real-time, while ResNet-18 excels in accuracy.',
        'For better results, ensure good lighting and centered objects.'
      ];
      insights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        insightsList.appendChild(li);
      });

      // ==================== AI DETECTION DISPLAY ====================
      if (data.ai_detection) {
        const aiCard = document.getElementById('aiDetectionCard');
        const aiImageType = document.getElementById('aiImageType');
        const aiConfidence = document.getElementById('aiConfidence');
        const aiMethod = document.getElementById('aiMethod');
        const aiVerdict = document.getElementById('aiVerdict');
        const aiTechnicalDetails = document.getElementById('aiTechnicalDetails');
        
        // Show AI detection card
        if (aiCard) {
          aiCard.style.display = 'block';
        }
        
        // Set image type label with color coding
        if (aiImageType) {
          aiImageType.textContent = data.ai_detection.label;
          if (data.ai_detection.is_ai) {
            aiImageType.className = 'font-bold text-lg px-6 py-2 rounded-xl shadow-sm bg-orange-100 text-orange-700 border border-orange-300';
          } else {
            aiImageType.className = 'font-bold text-lg px-6 py-2 rounded-xl shadow-sm bg-green-100 text-green-700 border border-green-300';
          }
        }
        
        // Set confidence with color based on value
        if (aiConfidence) {
          aiConfidence.textContent = data.ai_detection.confidence + '%';
          if (data.ai_detection.confidence > 80) {
            aiConfidence.className = 'font-bold text-2xl text-green-600';
          } else if (data.ai_detection.confidence > 60) {
            aiConfidence.className = 'font-bold text-2xl text-yellow-600';
          } else {
            aiConfidence.className = 'font-bold text-2xl text-orange-600';
          }
        }
        
        // Set detection method
        if (aiMethod) {
          aiMethod.textContent = data.ai_detection.method.toUpperCase();
        }
        
        // Set verdict with appropriate styling
        if (aiVerdict) {
          aiVerdict.textContent = data.ai_detection.verdict;
          if (data.ai_detection.is_ai) {
            aiVerdict.className = 'p-6 rounded-2xl text-base font-semibold leading-relaxed shadow-lg border-2 bg-orange-50 text-orange-800 border-orange-200';
          } else {
            aiVerdict.className = 'p-6 rounded-2xl text-base font-semibold leading-relaxed shadow-lg border-2 bg-green-50 text-green-800 border-green-200';
          }
        }
        
        // Display technical metrics if available
        if (aiTechnicalDetails && data.ai_detection.metrics) {
          let html = '<div class="space-y-1">';
          for (const [key, value] of Object.entries(data.ai_detection.metrics)) {
            const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            html += `
              <div class="flex justify-between text-xs">
                <span class="text-slate-600">${displayKey}:</span>
                <span class="font-semibold">${typeof value === 'number' ? value.toFixed(4) : value}</span>
              </div>
            `;
          }
          html += '</div>';
          aiTechnicalDetails.innerHTML = html;
        }
        
        console.log('[AI Detection]', data.ai_detection);
      }
      // ==================== END AI DETECTION DISPLAY ====================

      // Store scores for PDF
      localStorage.setItem('scores', JSON.stringify(data.scores));

    } catch (err) {
      alert('Error during analysis: ' + err.message);
    }
  });

  // Download PDF Report
  downloadReportBtn.addEventListener('click', async () => {
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text('Object Detection Report', 105, 20, { align: 'center' });

    // Add image
    const imgData = resultImage.src;
    doc.addImage(imgData, 'JPEG', 10, 30, 80, 60);

    // Add details
    doc.setFontSize(12);
    doc.text(`Detected Object: ${detectedObject.textContent}`, 10, 100);
    doc.text(`Confidence: ${detectedConfidence.textContent}`, 10, 110);
    doc.text(`Best Model: ${bestModel.textContent}`, 10, 120);
    doc.text(`Analysis Time: ${analysisTime.textContent}`, 10, 130);
    doc.text(`Models Evaluated: ${modelsEvaluated.textContent}`, 10, 140);

    // Add scores
    doc.text('Model Scores:', 10, 150);
    let y = 160;
    Object.entries(JSON.parse(localStorage.getItem('scores') || '{}')).forEach(([model, score]) => {
      doc.text(`${model}: ${score}%`, 10, y);
      y += 10;
    });

    // Add chart as image
    const chartImg = document.getElementById('chart').toDataURL('image/png');
    doc.addImage(chartImg, 'PNG', 10, y, 190, 80);
    y += 90;

    // Add summary and insights
    doc.text('Summary:', 10, y);
    y += 10;
    doc.text(summaryText.textContent, 10, y, { maxWidth: 190 });
    y += 30;
    doc.text('Insights:', 10, y);
    y += 10;
    Array.from(insightsList.children).forEach((li, i) => {
      doc.text(`${i+1}. ${li.textContent}`, 10, y);
      y += 10;
    });

    // Add timestamp
    doc.text(`Generated: ${new Date().toLocaleString()}`, 10, y + 10);

    doc.save('object_detection_report.pdf');
  });

  // Test another
  testAnotherBtn.addEventListener('click', () => {
    location.reload();
  });

  // Helper: Enable analyze button
  function enableAnalyzeBtn() {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('bg-gray-300', 'text-gray-500', 'cursor-not-allowed');
    analyzeBtn.classList.add('bg-blue-500', 'hover:bg-blue-600', 'text-white');
    analyzeBtn.classList.remove('hidden');
  }

  // Hide other options after selection
  function hideOtherOptions(mode) {
    if (mode === 'upload') {
      useWebcamBtn.classList.add('hidden');
      orDivider.classList.add('hidden');
      webcamContainer.classList.add('hidden');
    } else if (mode === 'webcam') {
      uploadContainer.classList.add('hidden');
      orDivider.classList.add('hidden');
      preview.classList.add('hidden');
      captureBtn.classList.remove('hidden');
      liveToggleBtn.classList.remove('hidden');
    }
  }

  // DataURL to File
  function dataURLToFile(dataurl, filename) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new File([u8arr], filename, { type: mime });
  }
});