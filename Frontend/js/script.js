// Scroll-based navbar highlighting
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('.navbar a');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= sectionTop && pageYOffset < sectionTop + sectionHeight) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
            link.classList.add('active');
        }
    });
});

// Scroll-triggered animations for all .scroll-animate sections
const animatedSections = document.querySelectorAll('.scroll-animate');

const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        } else {
            entry.target.classList.remove('visible'); // allow repeat
        }
    });
}, {
    threshold: 0.3
});

animatedSections.forEach(section => {
    sectionObserver.observe(section);
});

// Track selected stock market
let selectedMarket = null;

// DOM references
const marketButtons = document.querySelectorAll('.sm .btn');
const marketNameSpan = document.getElementById('market-name');
const strategyMarketNameSpan = document.getElementById('strategy-market-name');
const predictionForm = document.querySelector('.prediction-form');
const strategyForm = document.querySelector('.strategy-form');

// Function to block access if no market selected
function blockAccessIfNoMarket(e) {
    if (!selectedMarket) {
        e.preventDefault();
        alert("Please select a stock market in the 'Stock Markets' section before accessing this feature.");
        document.querySelector('#sm').scrollIntoView({ behavior: 'smooth' });
    }
}

// Block navbar links to prediction and strategy sections
const restrictedNavLinks = document.querySelectorAll('.navbar a[href="#pr"], .navbar a[href="#sr"]');
restrictedNavLinks.forEach(link => {
    link.addEventListener('click', blockAccessIfNoMarket);
});

// Handle market selection
marketButtons.forEach(button => {
    button.addEventListener('click', (e) => {
        e.preventDefault();

        selectedMarket = button.getAttribute('data-market');

        // Update market name in both sections
        if (marketNameSpan) marketNameSpan.textContent = selectedMarket;
        if (strategyMarketNameSpan) strategyMarketNameSpan.textContent = selectedMarket;

        // Highlight selected button
        marketButtons.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');

        // 🔓 Unlock prediction and strategy sections
        document.querySelectorAll('.locked-section').forEach(section => {
            section.classList.remove('locked-section');
            const lockMsg = section.querySelector('.section-lock-message');
            if (lockMsg) lockMsg.remove();
        });

        // Scroll to prediction section
        document.querySelector('#pr').scrollIntoView({ behavior: 'smooth' });
    });
});

// Prevent form submissions if no market is selected
function checkMarketSelection(e) {
    if (!selectedMarket) {
        e.preventDefault();
        alert("Please select a stock market in the 'Stock Markets' section before submitting.");
        document.querySelector('#sm').scrollIntoView({ behavior: 'smooth' });
    }
}

if (predictionForm) {
    predictionForm.addEventListener('submit', checkMarketSelection);
}

if (strategyForm) {
    strategyForm.addEventListener('submit', checkMarketSelection);
}

// Reset button functionality
const resetBtn = document.getElementById('reset-btn');

resetBtn.addEventListener('click', (e) => {
    e.preventDefault();

    // Clear selected market
    selectedMarket = null;

    // Reset titles
    if (marketNameSpan) marketNameSpan.textContent = "any market";
    if (strategyMarketNameSpan) strategyMarketNameSpan.textContent = "for any market";

    // Remove button highlights
    marketButtons.forEach(btn => btn.classList.remove('selected'));

    // Re-lock prediction and strategy sections
    document.querySelectorAll('#pr, #sr').forEach(section => {
        if (!section.classList.contains('locked-section')) {
            section.classList.add('locked-section');

            const msg = document.createElement('div');
            msg.className = 'section-lock-message';
            msg.textContent = 'Please select a stock market to access this section.';
            section.appendChild(msg);
        }
    });

    // Scroll to Home section
    document.querySelector('#home').scrollIntoView({ behavior: 'smooth' });
});
