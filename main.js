/**
 * TermSheet Feature Finder - Main JavaScript
 * 
 * Handles client-side functionality including:
 * - File uploads
 * - Email and chat access
 * - Data validation
 * - Visualization of validation results
 * - Tab navigation
 * - Data export
 */

document.addEventListener('DOMContentLoaded', function() {
    // App configuration
    const config = {
        apiBaseUrl: '/api',
        validStatusColor: '#28a745',   // green
        invalidStatusColor: '#dc3545', // red
        reviewStatusColor: '#ffc107'   // yellow
    };
    
    // Application state
    const state = {
        currentDocument: null,
        validationData: null
    };

    // Initialize the application
    function init() {
        initEventListeners();
        
        // Check if we're on the results page
        if (window.location.pathname.includes('result.html')) {
            initResultsPage();
        }
    }

    // Set up all event listeners
    function initEventListeners() {
        // Main action buttons
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) uploadBtn.addEventListener('click', handleUploadButtonClick);
        
        const emailBtn = document.getElementById('emailBtn');
        if (emailBtn) emailBtn.addEventListener('click', handleEmailButtonClick);
        
        const chatBtn = document.getElementById('chatBtn');
        if (chatBtn) chatBtn.addEventListener('click', handleChatButtonClick);
        
        // Export button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) exportBtn.addEventListener('click', exportValidationData);
        
        // File upload handling
        initializeFileUpload();
        
        // Form submissions
        const emailForm = document.getElementById('emailForm');
        if (emailForm) emailForm.addEventListener('submit', handleEmailFormSubmit);
        
        const chatForm = document.getElementById('chatForm');
        if (chatForm) chatForm.addEventListener('submit', handleChatFormSubmit);
        
        // Phone number input for OTP flow
        const phoneNumberInput = document.getElementById('phoneNumber');
        if (phoneNumberInput) {
            phoneNumberInput.addEventListener('input', handlePhoneNumberInput);
        }
        
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(button => {
            button.addEventListener('click', handleTabClick);
        });
    }

    // Handle upload button click
    function handleUploadButtonClick(e) {
        e.preventDefault();
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.click();
        } else {
            // If we're not on the upload page, redirect
            window.location.href = 'index.html#uploadSection';
        }
    }

    // Handle email button click
    function handleEmailButtonClick(e) {
        e.preventDefault();
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('chatSection').style.display = 'none';
        document.getElementById('emailSection').style.display = 'block';
        document.getElementById('emailSection').scrollIntoView({ behavior: 'smooth' });
    }

    // Handle chat button click
    function handleChatButtonClick(e) {
        e.preventDefault();
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('emailSection').style.display = 'none';
        document.getElementById('chatSection').style.display = 'block';
        document.getElementById('chatSection').scrollIntoView({ behavior: 'smooth' });
    }

    // Handle phone number input for OTP flow
    function handlePhoneNumberInput(e) {
        const phoneNumber = e.target.value;
        if (phoneNumber.length >= 10) {
            document.getElementById('otpGroup').style.display = 'block';
        } else {
            document.getElementById('otpGroup').style.display = 'none';
        }
    }

    // Initialize file upload functionality
    function initializeFileUpload() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        if (!dropZone || !fileInput) return;
        
        // Click on drop zone opens file selection
        dropZone.addEventListener('click', () => fileInput.click());
        
        // Handle file selection
        fileInput.addEventListener('change', function() { 
            if (this.files?.[0]) handleFileSelection(this.files[0]); 
        });
        
        // Prevent default behaviors for drag events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        // Add visual feedback for drag actions
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('active'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('active'), false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', (e) => {
            const file = e.dataTransfer.files[0];
            if (file) handleFileSelection(file);
        }, false);
    }

    // Process the selected file
    function handleFileSelection(file) {
        const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        // Validate file type
        if (!validTypes.includes(file.type) && !['pdf', 'docx'].includes(fileExtension)) {
            showNotification('Invalid file type. Please upload a PDF or DOCX file.', 'error');
            return;
        }
        
        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            showNotification('File size exceeds 10MB limit.', 'error');
            return;
        }
        
        state.currentDocument = file;
        showLoading('Processing your TermSheet...');
        
        // Upload file to server
        const formData = new FormData();
        formData.append('file', file);
        
        fetch(`${config.apiBaseUrl}/upload`, { 
            method: 'POST', 
            body: formData 
        })
        .then(handleResponse)
        .then(data => {
            if (data.success) {
                // Store the result data and navigate to results page
                localStorage.setItem('termsheetData', JSON.stringify(data));
                window.location.href = 'result.html';
            } else {
                throw new Error(data.error || 'Failed to process file');
            }
        })
        .catch(error => {
            hideLoading();
            showNotification(error.message || 'File processing error', 'error');
        });
    }

    // Handle email form submission
    function handleEmailFormSubmit(e) {
        e.preventDefault();
        const email = document.getElementById('emailAddress').value;
        const password = document.getElementById('password').value;
        const termsheetId = document.getElementById('termsheetId').value;
        
        if (!email || !password || !termsheetId) {
            showNotification('Please fill all required fields', 'error');
            return;
        }
        
        showLoading('Accessing termsheet from email...');
        
        fetch(`${config.apiBaseUrl}/email-access`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email: email,
                password: password,
                termsheetId: termsheetId
            })
        })
        .then(handleResponse)
        .then(data => {
            if (data.success) {
                localStorage.setItem('termsheetData', JSON.stringify(data.data));
                window.location.href = 'result.html';
            } else {
                throw new Error(data.error || 'Failed to access termsheet from email');
            }
        })
        .catch(error => {
            hideLoading();
            showNotification(error.message || 'Email access error', 'error');
        });
    }

    // Handle chat form submission
    function handleChatFormSubmit(e) {
        e.preventDefault();
        const phoneNumber = document.getElementById('phoneNumber').value;
        const otp = document.getElementById('otp').value;
        const termsheetId = document.getElementById('chatTermsheetId').value;
        
        if (!phoneNumber || !termsheetId) {
            showNotification('Phone number and termsheet ID are required', 'error');
            return;
        }
        
        if (document.getElementById('otpGroup').style.display !== 'none' && !otp) {
            showNotification('OTP is required', 'error');
            return;
        }
        
        showLoading('Accessing termsheet from chat...');
        
        fetch(`${config.apiBaseUrl}/chat-access`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                phone: phoneNumber,
                otp: otp,
                termsheetId: termsheetId
            })
        })
        .then(handleResponse)
        .then(data => {
            if (data.success) {
                localStorage.setItem('termsheetData', JSON.stringify(data.data));
                window.location.href = 'result.html';
            } else {
                throw new Error(data.error || 'Failed to access termsheet from chat');
            }
        })
        .catch(error => {
            hideLoading();
            showNotification(error.message || 'Chat access error', 'error');
        });
    }

    // Initialize the results page with data from localStorage
    function initResultsPage() {
        // Retrieve termsheet data from localStorage
        const termsheetData = JSON.parse(localStorage.getItem('termsheetData') || '{}');
        
        if (Object.keys(termsheetData).length === 0) {
            showNotification('No data available. Please process a termsheet first.', 'error');
            return;
        }

        // Extract data from potentially different response structures
        let extractedData = termsheetData.extracted_data || {};
        let validationResults = termsheetData.validation_results || {};

        // Handle case where data is nested
        if (termsheetData.data) {
            extractedData = termsheetData.data.extracted_data || extractedData;
            validationResults = termsheetData.data.validation_results || validationResults;
        }

        // Populate document information
        populateDocumentInfo(extractedData);

        // Process and display validation data
        if (validationResults && Object.keys(validationResults).length > 0) {
            const validationItems = transformValidationData(validationResults, extractedData);
            state.validationData = validationItems;
            
            populateValidationTable(validationItems);
            createValidationDonutChart(validationItems);
        } else {
            showEmptyValidationTable();
        }
    }

    // Populate the document info fields with extracted data
    function populateDocumentInfo(data) {
        // Update client and issuer information
        const issuerElement = document.querySelector('.Issuer');
        const clientElement = document.getElementById('Client');
        
        if (issuerElement) {
            issuerElement.textContent = data.issuer || 'N/A';
        }
        if (clientElement) {
            clientElement.textContent = data.client || 'N/A';
        }
        
        // Update key dates
        updateElementText('TradeDate', formatDate(data.trade_date));
        updateElementText('valueDate', formatDate(data.value_date || data.trade_date)); // Fallback to trade_date if value_date not found
        updateElementText('ExpiryDate', formatDate(data.expiry_date));
        updateElementText('DeliveryDate', formatDate(data.delivery_date));
        
        // Update financial terms
        updateElementText('Reference_Spot_Price', formatCurrency(data.ref_spot_price));
        updateElementText('Notional_Amount', formatCurrency(data.notional_amount));
        updateElementText('Strike_Price', formatCurrency(data.strike_price));
        updateElementText('Premium_Rate', data.premium_rate ? data.premium_rate + '%' : 'N/A');
        
        // Update option details
        updateElementText('Option_Type', data.option_type);
        updateElementText('Business_Calendar', data.business_calendar);
        updateElementText('Settlements_Type', data.settlement_type || data.settlements_type || 'N/A');
        updateElementText('Settlements_Method', data.settlement_method || data.settlements_method || 'N/A');
        
        // Update currency information
        updateElementText('Transaction_Currency', data.transaction_ccy || data.transaction_currency || 'N/A');
        updateElementText('Counter_Currency', data.counter_ccy || data.counter_currency || 'N/A');
    }

    // Create the validation donut chart with Chart.js
    function createValidationDonutChart(validationItems) {
        const ctx = document.getElementById('validationDonutChart');
        if (!ctx) return;

        // Count status occurrences
        const statusCounts = {
            'Valid': 0,
            'Invalid': 0,
            'Needs Review': 0
        };

        // Count items by status
        validationItems.forEach(item => {
            if (item.status === 'Valid') statusCounts['Valid']++;
            else if (item.status === 'Invalid') statusCounts['Invalid']++;
            else statusCounts['Needs Review']++;
        });

        // Update the counts in the legend
        document.getElementById('validCount').textContent = statusCounts['Valid'];
        document.getElementById('invalidCount').textContent = statusCounts['Invalid'];
        document.getElementById('reviewCount').textContent = statusCounts['Needs Review'];

        // Destroy existing chart instance if it exists
        if (window.validationChart instanceof Chart) {
            window.validationChart.destroy();
        }

        // Create the donut chart
        window.validationChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Valid', 'Invalid', 'Needs Review'],
                datasets: [{
                    data: [statusCounts['Valid'], statusCounts['Invalid'], statusCounts['Needs Review']],
                    backgroundColor: [
                        config.validStatusColor,
                        config.invalidStatusColor,
                        config.reviewStatusColor
                    ],
                    borderWidth: 1,
                    cutout: '70%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Transform raw validation data to a standardized format
    function transformValidationData(validationResults, extractedData) {
        const validationData = validationResults.validation_results || validationResults;
    
        if (!validationData || typeof validationData !== 'object') {
            console.warn('Invalid validation data format:', validationData);
            return [];
        }
    
        return Object.entries(validationData).map(([param, info]) => {
            const extracted = extractedData[param] || 'N/A';
            const masterList = Array.isArray(info.master_list) 
                ? info.master_list.join(', ') 
                : (info.master_values ? info.master_values.join(', ') : '—');
    
            // Determine status based on the data structure
            let status;
            if (info.status) {
                status = info.status;
            } else if (info.match === true) {
                status = 'Valid';
            } else if (info.match === false) {
                status = 'Invalid';
            } else {
                status = 'Needs Review';
            }
    
            const comment = info.comment || '—';
    
            return {
                parameter: param,
                value: extracted,
                master: masterList,
                status: status,
                comment: comment
            };
        });
    }
    
    // Populate validation table with data
    function populateValidationTable(validationItems) {
        const tbody = document.querySelector('#validation-table tbody');
        if (!tbody) return;
    
        tbody.innerHTML = '';
    
        if (!validationItems || validationItems.length === 0) {
            showEmptyValidationTable();
            return;
        }
    
        validationItems.forEach(item => {
            const row = document.createElement('tr');
    
            // Parameter cell
            const paramCell = document.createElement('td');
            paramCell.textContent = formatParameterName(item.parameter);
            row.appendChild(paramCell);
    
            // Extracted Value cell
            const valueCell = document.createElement('td');
            valueCell.textContent = item.value || 'N/A';
            row.appendChild(valueCell);
    
            // Master Values cell
            const masterCell = document.createElement('td');
            masterCell.textContent = item.master || '—';
            row.appendChild(masterCell);
    
            // Status cell with color coding
            const statusCell = document.createElement('td');
            statusCell.textContent = item.status;
    
            // Apply appropriate class based on status
            if (item.status === 'Valid') {
                statusCell.classList.add('status-valid');
            } else if (item.status === 'Invalid') {
                statusCell.classList.add('status-invalid');
            } else {
                statusCell.classList.add('status-review');
            }
            row.appendChild(statusCell);
    
            // Comment cell
            const commentCell = document.createElement('td');
            commentCell.textContent = item.comment || '—';
            commentCell.classList.add('comment');
            row.appendChild(commentCell);
    
            tbody.appendChild(row);
        });
    }
    
    // Display empty validation table with message
    function showEmptyValidationTable() {
        const tbody = document.querySelector('#validation-table tbody');
        if (!tbody) return;
        
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 5;
        cell.textContent = 'No validation data available';
        cell.style.textAlign = 'center';
        row.appendChild(cell);
        tbody.appendChild(row);
    }

    // Handle tab navigation
    function handleTabClick() {
        const tabName = this.getAttribute('data-tab');
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');
        
        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(tabName + 'Tab').classList.add('active');
    }

    // Export validation data to CSV
    function exportValidationData() {
        if (!state.validationData || state.validationData.length === 0) {
            const termsheetData = JSON.parse(localStorage.getItem('termsheetData') || '{}');
            let validationResults = termsheetData.validation_results || {};
            
            if (termsheetData.data) {
                validationResults = termsheetData.data.validation_results || validationResults;
            }
            
            if (!validationResults || Object.keys(validationResults).length === 0) {
                showNotification('No validation data available to export', 'error');
                return;
            }
            
            const extractedData = termsheetData.extracted_data || 
                                 (termsheetData.data ? termsheetData.data.extracted_data : {});
            state.validationData = transformValidationData(validationResults, extractedData);
        }
        
        // Create CSV content
        let csvContent = 'Parameter,Extracted Value,Master Sheet Value(s),Status,Comment\n';
        
        state.validationData.forEach(item => {
            const escapedParam = escapeCSVValue(formatParameterName(item.parameter));
            const escapedValue = escapeCSVValue(item.value);
            const escapedMaster = escapeCSVValue(item.master);
            const escapedStatus = escapeCSVValue(item.status);
            const escapedComment = escapeCSVValue(item.comment);
            
            csvContent += `${escapedParam},${escapedValue},${escapedMaster},${escapedStatus},${escapedComment}\n`;
        });
        
        // Generate download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `termsheet_validation_${formatDateForFilename(new Date())}.csv`;
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Clean up
        setTimeout(() => {
            URL.revokeObjectURL(url);
            document.body.removeChild(link);
        }, 100);
    }

    // Helper function to format date for filename
    function formatDateForFilename(date) {
        return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}_${String(date.getHours()).padStart(2, '0')}-${String(date.getMinutes()).padStart(2, '0')}`;
    }

    // Helper function to escape CSV values
    function escapeCSVValue(value) {
        if (value === null || value === undefined) return '""';
        return `"${String(value).replace(/"/g, '""')}"`;
    }

    // Helper function to update element text content
    function updateElementText(selector, value) {
        const element = selector.startsWith('.') 
            ? document.querySelector(selector)
            : document.getElementById(selector);
        if (element) element.textContent = value || 'N/A';
    }

    // Helper function to format parameter names
    function formatParameterName(param) {
        return param?.replace(/_/g, ' ').replace(/(^|\s)\w/g, match => match.toUpperCase()) || '';
    }

    // Helper function to format date
    function formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
        } catch (error) {
            return dateString || 'N/A';
        }
    }

    // Helper function to format currency
    function formatCurrency(value, currency = 'USD') {
        if (!value && value !== 0) return 'N/A';
        try {
            return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(value);
        } catch (error) {
            return value?.toString() || 'N/A';
        }
    }

    // Helper function to prevent default behavior
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Helper function to show loading overlay
    function showLoading(message) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            const messageElement = loadingOverlay.querySelector('p');
            if (messageElement) messageElement.textContent = message || 'Processing...';
            loadingOverlay.style.display = 'flex';
        }
    }

    // Helper function to hide loading overlay
    function hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) loadingOverlay.style.display = 'none';
    }

    // Helper function to show notification
    function showNotification(message, type = 'info') {
        // Simple alert for now - could be enhanced with a custom notification system
        alert(message);
    }

    // Helper function to handle API responses
    function handleResponse(response) {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Request failed');
            });
        }
        return response.json();
    }

    // Initialize the application
    init();
});