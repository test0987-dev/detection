document.addEventListener('DOMContentLoaded', function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const resultsList = document.getElementById('resultsList');
    
    // Handle drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Handle click on dropzone
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });
    
    function handleFileUpload(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a valid image (JPEG, PNG, GIF)');
            return;
        }
    
        // Check file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            showError('File size too large (max 10MB)');
            return;
        }
    
        showLoading();
        
        // Preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);
    
        // Create FormData and send
        const formData = new FormData();
        formData.append('file', file);
    
        fetch('/predict', {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error') });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            console.error('Upload error:', error);
            showError(error.message || 'Failed to process image');
        })
        .finally(() => {
            hideLoading();
        });
    }
    
    function displayResults(detections) {
        resultsList.innerHTML = '';
        
        if (detections.length === 0) {
            resultsList.innerHTML = '<div class="detection-item">No wildlife detected in the image</div>';
            return;
        }
        
        detections.forEach(detection => {
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
            
            const confidencePercent = Math.round(detection.confidence * 100);
            
            detectionItem.innerHTML = `
                <h4>${detection.class}</h4>
                <p>Confidence: ${confidencePercent}%</p>
                <p>Bounding Box: [${detection.bbox.map(x => x.toFixed(1)).join(', ')}]</p>
                <div class="confidence-bar">
                    <div class="confidence-level" style="width: ${confidencePercent}%"></div>
                </div>
            `;
            
            resultsList.appendChild(detectionItem);
        });
    }
    
    function showError(message) {
        resultsList.innerHTML = `<div class="detection-item error">${message}</div>`;
    }
    
    function showLoading() {
        resultsList.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing image...</p>
            </div>
        `;
    }
    
    function hideLoading() {
        const loadingElement = document.querySelector('.loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
    }
});