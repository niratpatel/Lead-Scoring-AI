document.addEventListener('DOMContentLoaded', () => {
const API_URL = 'https://lead-scoring-ai.onrender.com'; 
    const form = document.getElementById('lead-form');
    const tableBody = document.getElementById('leads-table-body');
    const submitBtn = document.getElementById('submit-btn');
    const creditScoreSlider = document.getElementById('credit_score');
    const creditScoreValueSpan = document.getElementById('credit_score_value');
    const sortBtn = document.getElementById('sort-btn');
    
    let allLeads = []; // Our local, in-memory store for leads
    let scoreChartInstance = null; // To hold the chart object

    // --- HELPER FUNCTIONS ---
    const formatScore = (score) => {
        const num = Number(score);
        return isNaN(num) ? '0.00' : num.toFixed(2);
    };

    const saveLeadsToLocal = () => {
        localStorage.setItem('scoredLeads', JSON.stringify(allLeads));
    };

    const loadLeadsFromLocal = () => {
        const storedLeads = localStorage.getItem('scoredLeads');
        return storedLeads ? JSON.parse(storedLeads) : [];
    };

    // --- UI RENDERING FUNCTIONS ---
    const renderTable = (leads) => {
        tableBody.innerHTML = '';
        if (leads.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;">No leads scored yet.</td></tr>';
            return;
        }
        leads.forEach(lead => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${lead.email}</td><td>${lead.comments || 'N/A'}</td><td>${formatScore(lead.initial_score)}</td><td><b>${formatScore(lead.reranked_score)}</b></td>`;
            tableBody.appendChild(row);
        });
    };

    const updateScoreChart = (leads) => {
        const scoreCategories = { 'Nurture (0-39)': 0, 'Cold (40-59)': 0, 'Warm (60-79)': 0, 'Hot (80-100)': 0 };
        leads.forEach(lead => {
            const score = lead.reranked_score;
            if (score >= 80) scoreCategories['Hot (80-100)']++;
            else if (score >= 60) scoreCategories['Warm (60-79)']++;
            else if (score >= 40) scoreCategories['Cold (40-59)']++;
            else scoreCategories['Nurture (0-39)']++;
        });

        const chartData = {
            labels: Object.keys(scoreCategories),
            datasets: [{
                label: 'Number of Leads',
                data: Object.values(scoreCategories),
                backgroundColor: ['#3498db', '#f1c40f', '#e67e22', '#e74c3c'],
                borderColor: '#ffffff',
                borderWidth: 1
            }]
        };

        if (scoreChartInstance) {
            scoreChartInstance.data = chartData;
            scoreChartInstance.update();
        } else {
            const ctx = document.getElementById('scoreChart').getContext('2d');
            scoreChartInstance = new Chart(ctx, {
                type: 'bar',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: { y: { beginAtZero: true } },
                    plugins: { legend: { display: false } }
                }
            });
        }
    };

    const renderUI = (leads) => {
        renderTable(leads);
        updateScoreChart(leads);
    };

    // --- EVENT LISTENERS ---
    creditScoreSlider.addEventListener('input', () => {
        creditScoreValueSpan.textContent = creditScoreSlider.value;
    });

    sortBtn.addEventListener('click', () => {
        allLeads.sort((a, b) => b.reranked_score - a.reranked_score);
        renderUI(allLeads); // Re-render the UI with the sorted data
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = { /* This part is unchanged */
            email: document.getElementById('email').value, page_views: parseInt(document.getElementById('page_views').value),
            form_submissions: parseInt(document.getElementById('form_submissions').value), previous_interactions: parseInt(document.getElementById('previous_interactions').value),
            time_spent_website_secs: parseInt(document.getElementById('time_spent_website_secs').value), credit_score: parseInt(document.getElementById('credit_score').value),
            income_inr: parseInt(document.getElementById('income_inr').value), profession: document.getElementById('profession').value,
            city: document.getElementById('city').value, property_type_preference: document.getElementById('property_type_preference').value,
            lead_source: document.getElementById('lead_source').value, gender: document.getElementById('gender').value,
            age_group: document.getElementById('age_group').value, education: document.getElementById('education').value,
            family_background: document.getElementById('family_background').value, comments: document.getElementById('comments').value
        };
        if (!document.getElementById('consent').checked) { alert('You must consent to data processing.'); return; }
        submitBtn.textContent = 'Scoring...'; submitBtn.disabled = true;

        try {
            const response = await fetch(`${API_URL}/score`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(formData) });
            const newLead = await response.json();
            if (!response.ok || newLead.error) { throw new Error(newLead.detail || newLead.error || `HTTP error`); }

            allLeads.unshift(newLead); // Add the new lead to the beginning of our local array
            saveLeadsToLocal();         // Persist the updated array to localStorage
            renderUI(allLeads);         // Re-render the entire UI with the new data
            
            form.reset();
            creditScoreValueSpan.textContent = "650";
        } catch (error) {
            console.error('Error submitting lead:', error);
            alert(`Failed to score lead: ${error.message}`);
        } finally {
            submitBtn.textContent = 'Get Accurate Score';
            submitBtn.disabled = false;
        }
    });

    // --- INITIAL PAGE LOAD ---
    const initializePage = () => {
        allLeads = loadLeadsFromLocal();
        renderUI(allLeads);
    };

    initializePage();
});