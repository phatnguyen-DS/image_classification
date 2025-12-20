document.addEventListener('DOMContentLoaded', function () {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const hamburger = document.querySelector('.hamburger');
    const nav = document.querySelector('.nav');

    // Mobile menu toggle
    hamburger.addEventListener('click', function () {
        nav.classList.toggle('active');
        hamburger.classList.toggle('active');
    });

    // Smooth scrolling
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);

            if (targetSection) {
                window.scrollTo({
                    top: targetSection.offsetTop - 80,
                    behavior: 'smooth'
                });
            }

            // Close mobile menu if open
            nav.classList.remove('active');
            hamburger.classList.remove('active');

            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Update active nav link on scroll
    window.addEventListener('scroll', function () {
        let current = '';
        const sections = document.querySelectorAll('section');

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (scrollY >= (sectionTop - 100)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });

    // File upload and image analysis
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const resultSection = document.getElementById('resultSection');
    const loading = document.getElementById('loading');
    const resultContent = document.getElementById('resultContent');
    const emptyState = document.getElementById('emptyState');
    const resultCard = document.getElementById('resultCard');
    const diagnosisResult = document.getElementById('diagnosisResult');
    const progressFill = document.getElementById('progressFill');
    const confidenceValue = document.getElementById('confidenceValue');

    // Click to upload
    uploadArea.addEventListener('click', function () {
        fileInput.click();
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        this.style.backgroundColor = 'rgba(44, 123, 229, 0.1)';
    });

    uploadArea.addEventListener('dragleave', function (e) {
        e.preventDefault();
        this.style.backgroundColor = '';
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        this.style.backgroundColor = '';

        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            showPreview();
        }
    });

    // File input change
    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            showPreview();
        }
    });

    // Show image preview
    function showPreview() {
        const file = fileInput.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                previewImage.src = e.target.result;
                uploadArea.style.display = 'none';
                previewContainer.style.display = 'block';
                // Keep result section visible but in empty state or previous result
                if (resultContent.style.display !== 'block') {
                    emptyState.style.display = 'flex';
                    resultContent.style.display = 'none';
                }
            };

            reader.readAsDataURL(file);
        }
    }

    // Clear image
    clearBtn.addEventListener('click', function () {
        fileInput.value = '';
        uploadArea.style.display = 'block';
        previewContainer.style.display = 'none';

        // Reset Result Section to Empty State
        emptyState.style.display = 'flex';
        loading.style.display = 'none';
        resultContent.style.display = 'none';
    });

    // Analyze button
    analyzeBtn.addEventListener('click', async function () {
        if (!fileInput.files.length) {
            showNotification('Vui lòng chọn một hình ảnh!', 'warning');
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        // Show loading
        emptyState.style.display = 'none';
        loading.style.display = 'flex';
        resultContent.style.display = 'none';
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Đang phân tích...';

        try {
            // Simulate API call (replace with actual API call)
            simulateAPICall();

            // Uncomment the following code for actual API call
            /*
            const response = await fetch('https://e2e-vision-pipeline-onnx-backend.onrender.com/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Lỗi khi gọi API. Vui lòng thử lại sau.');
            }
            
            const data = await response.json();
            displayResult(data);
            */

        } catch (error) {
            console.error('Error:', error);
            showNotification(error.message || 'Đã xảy ra lỗi. Vui lòng thử lại sau!', 'danger');
            loading.style.display = 'none';
            emptyState.style.display = 'flex';
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Phân tích';
        }
    });

    // Simulate API call with sample data
    function simulateAPICall() {
        setTimeout(() => {
            const sampleData = {
                predictions: [
                    { class: 'Eczema', confidence: 0.85 },
                    { class: 'Psoriasis', confidence: 0.12 },
                    { class: 'Acne', confidence: 0.03 }
                ]
            };
            displayResult(sampleData);
        }, 2000);
    }

    // Display analysis result
    function displayResult(data) {
        loading.style.display = 'none';
        resultContent.style.display = 'block';

        let html = '';

        if (data.predictions && Array.isArray(data.predictions)) {
            // Display top 3 predictions
            const topPredictions = data.predictions.slice(0, 3);

            topPredictions.forEach((prediction, index) => {
                const className = prediction.class || 'Unknown';
                const confidence = prediction.confidence || 0;
                const confidencePercent = Math.round(confidence * 100);

                // Add highlight to top prediction
                const isTopPrediction = index === 0;
                const highlightClass = isTopPrediction ? 'top-prediction' : '';

                html += `
                    <div class="diagnosis-item ${highlightClass}">
                        <span class="diagnosis-name">
                            ${isTopPrediction ? '<i class="fas fa-star"></i> ' : ''}${className}
                        </span>
                        <span class="diagnosis-confidence">${confidencePercent}%</span>
                    </div>
                `;
            });

            // Animate progress bar for top prediction
            setTimeout(() => {
                const topConfidence = data.predictions[0].confidence;
                progressFill.style.width = `${topConfidence * 100}%`;
                confidenceValue.textContent = `${Math.round(topConfidence * 100)}%`;
            }, 300);

        } else if (data.prediction) {
            // Single prediction result
            const className = data.prediction;
            const confidence = data.confidence || 0.8; // Default confidence if not provided

            html = `
                <div class="diagnosis-item top-prediction">
                    <span class="diagnosis-name">
                        <i class="fas fa-star"></i> ${className}
                    </span>
                    <span class="diagnosis-confidence">${Math.round(confidence * 100)}%</span>
                </div>
            `;

            // Animate progress bar
            setTimeout(() => {
                progressFill.style.width = `${confidence * 100}%`;
                confidenceValue.textContent = `${Math.round(confidence * 100)}%`;
            }, 300);
        } else {
            // Display raw response if format is unexpected
            html = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }

        diagnosisResult.innerHTML = html;
    }

    // Show notification
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close"><i class="fas fa-times"></i></button>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Show with animation
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);

        // Auto remove after 5 seconds
        setTimeout(() => {
            removeNotification(notification);
        }, 5000);

        // Close button
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            removeNotification(notification);
        });
    }

    function removeNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    function getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'fa-check-circle';
            case 'warning': return 'fa-exclamation-triangle';
            case 'danger': return 'fa-times-circle';
            case 'info':
            default: return 'fa-info-circle';
        }
    }

    // Add notification styles
    const notificationStyles = document.createElement('style');
    notificationStyles.innerHTML = `
        .notification {
            position: fixed;
            top: 20px;
            right: -400px;
            z-index: 9999;
            max-width: 350px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            overflow: hidden;
            transition: right 0.3s ease;
        }
        
        .notification.show {
            right: 20px;
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            padding: 15px;
        }
        
        .notification-content i {
            margin-right: 10px;
            font-size: 1.2rem;
        }
        
        .notification-content span {
            flex-grow: 1;
        }
        
        .notification-close {
            background: none;
            border: none;
            margin-left: 10px;
            cursor: pointer;
            color: #8898aa;
        }
        
        .notification-success i {
            color: #2dce89;
        }
        
        .notification-warning i {
            color: #fb6340;
        }
        
        .notification-danger i {
            color: #f5365c;
        }
        
        .notification-info i {
            color: #11cdef;
        }
        
        .diagnosis-item.top-prediction {
            background-color: rgba(44, 123, 229, 0.05);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .diagnosis-item.top-prediction .diagnosis-name {
            color: #2c7be5;
            font-weight: 700;
        }
        
        .diagnosis-item.top-prediction .diagnosis-confidence {
            font-weight: 700;
        }
        
        .diagnosis-item.top-prediction i {
            margin-right: 8px;
        }
    `;

    document.head.appendChild(notificationStyles);
});