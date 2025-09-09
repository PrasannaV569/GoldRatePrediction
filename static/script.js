// Animate cards on page load
window.addEventListener('load', () => {
    const cards = document.querySelectorAll('.animate-card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 300);
    });

    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            refreshBtn.disabled = true;
            refreshBtn.textContent = 'Refreshing...';
            window.location.reload();
        });
    }

    // Theme toggle
    const themeToggle = document.getElementById('themeToggle');
    const storedTheme = localStorage.getItem('theme') || 'dark';
    if (storedTheme === 'light') {
        document.documentElement.classList.add('theme-light');
    }
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const isLight = document.documentElement.classList.toggle('theme-light');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
        });
    }
});

// Live date & time
function updateDateTime() {
    const dateTimeEl = document.getElementById('dateTime');
    const now = new Date();
    const formatted = now.toLocaleString('en-IN', { hour12: false });
    dateTimeEl.textContent = 'Date & Time: ' + formatted;
}
setInterval(updateDateTime, 1000);

// Graceful initial call
updateDateTime();
