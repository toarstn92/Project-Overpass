document.addEventListener('DOMContentLoaded', () => {
    let cards = document.querySelectorAll('.nhsuk-card--clickable')
    document.querySelectorAll('.nhsuk-card--clickable').forEach((card) => {
        if (card.querySelector('a') !== null) {
            // Clicks the link within the heading to navigate to desired page
            card.addEventListener('click', () => {
            card.querySelector('a').click();
            });
        }
    });
});
